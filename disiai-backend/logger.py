"""Evaluation logging: every recommendation and the user's decision."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def init_evaluation_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            detected_context TEXT NOT NULL,
            recommended_modality TEXT NOT NULL,
            user_decision TEXT NOT NULL,
            switching_latency_ms INTEGER NOT NULL DEFAULT 0,
            target_modality TEXT,
            reason TEXT,
            confidence REAL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_conversation ON evaluation_log(conversation_id)"
    )
    connection.commit()


def log_evaluation(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    detected_context: dict[str, Any],
    recommended_modality: str,
    user_decision: str,
    switching_latency_ms: int,
    target_modality: str | None = None,
    reason: str = "",
    confidence: float = 0.0,
) -> int:
    """Insert one transaction into the evaluation log and return its id."""
    init_evaluation_table(connection)
    cursor = connection.execute(
        """
        INSERT INTO evaluation_log (
            timestamp, conversation_id, detected_context, recommended_modality,
            user_decision, switching_latency_ms, target_modality, reason, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            conversation_id,
            json.dumps(detected_context, ensure_ascii=False),
            recommended_modality,
            user_decision,
            int(max(0, switching_latency_ms)),
            target_modality,
            reason,
            float(confidence),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def get_evaluation_logs(
    connection: sqlite3.Connection,
    *,
    conversation_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    init_evaluation_table(connection)
    query = "SELECT * FROM evaluation_log"
    params: list[Any] = []
    if conversation_id:
        query += " WHERE conversation_id = ?"
        params.append(conversation_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = connection.execute(query, params).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["detected_context"] = json.loads(item["detected_context"])
        results.append(item)
    return results
