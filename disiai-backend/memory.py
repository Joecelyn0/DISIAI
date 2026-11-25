"""SQLite-backed conversation memory, preferences and context handoff."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from config import settings
from logger import init_evaluation_table


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(settings.db_path, timeout=10.0, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        yield connection
    finally:
        connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                modality TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id);

            CREATE TABLE IF NOT EXISTS preferences (
                conversation_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (conversation_id, modality)
            );
            """
        )
        init_evaluation_table(connection)


def touch_conversation(conversation_id: str) -> None:
    now = _now()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (conversation_id, now, now),
        )
        connection.commit()


def save_message(
    *,
    conversation_id: str,
    role: str,
    modality: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    touch_conversation(conversation_id)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, modality, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                modality,
                content,
                json.dumps(metadata or {}, ensure_ascii=False),
                _now(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_message(message_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["metadata"] = json.loads(item["metadata"])
    return item


def get_messages(conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM messages WHERE conversation_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    results = []
    for row in reversed(rows):
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"])
        results.append(item)
    return results


def get_preference(conversation_id: str) -> str | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT modality FROM preferences WHERE conversation_id = ?
            ORDER BY count DESC, updated_at DESC LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
    return row["modality"] if row else None


def record_preference(conversation_id: str, modality: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO preferences (conversation_id, modality, count, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(conversation_id, modality) DO UPDATE SET
                count = count + 1,
                updated_at = excluded.updated_at
            """,
            (conversation_id, modality, _now()),
        )
        connection.commit()


def build_context_summary(
    conversation_id: str,
    *,
    max_messages: int = 12,
    exclude_message_id: int | None = None,
) -> str:
    """Compose a compact handoff summary for a newly selected modality."""
    messages = get_messages(conversation_id, limit=max_messages)
    if exclude_message_id is not None:
        messages = [m for m in messages if m["id"] != exclude_message_id]
    if not messages:
        return "Belum ada riwayat percakapan pada conversation_id ini."

    lines = []
    for message in messages:
        content = " ".join(message["content"].split())
        if len(content) > 300:
            content = content[:297] + "..."
        lines.append(f"- {message['role']} ({message['modality']}): {content}")
    return "Ringkasan percakapan sebelumnya:\n" + "\n".join(lines)
