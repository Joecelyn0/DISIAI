"""DISIAI FastAPI middleware: adaptive multimodal routing with human confirmation."""

import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from engine import DetectedContext, Modality, Recommendation, recommend
from extractor import extract_context
from handlers import avatar_sadtalker, text_qwen, voice_whisper
from logger import get_evaluation_logs, log_evaluation
from memory import (
    build_context_summary,
    get_message,
    get_preference,
    init_db,
    record_preference,
    save_message,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.static_dir.mkdir(parents=True, exist_ok=True)
    (settings.static_dir / "mock" / "out").mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=settings.static_dir), name="media")


class ExtractRequest(BaseModel):
    conversation_id: str
    text: str
    use_llm: bool | None = None


class RecommendRequest(BaseModel):
    conversation_id: str
    text: str = Field(min_length=1)
    input_modality: Modality = Modality.TEXT
    use_llm: bool | None = None


class ConfirmRequest(BaseModel):
    conversation_id: str
    message_id: int
    decision: Literal["accept", "reject"]
    target_modality: Modality | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "env": settings.env,
        "ollama_reachable": text_qwen.is_available(),
        "ollama_model": settings.ollama_model,
        "database": str(settings.db_path),
        "database_ready": settings.db_path.exists(),
        "mock_video_ready": settings.resolve_path(settings.mock_video_path).exists(),
    }


@app.post("/context/extract")
def extract(request: ExtractRequest) -> dict[str, Any]:
    preference = get_preference(request.conversation_id)
    context = extract_context(
        request.text,
        historical_preference=preference,
        use_llm=request.use_llm,
    )
    return context.as_dict()


@app.post("/recommend")
def recommend_endpoint(request: RecommendRequest) -> dict[str, Any]:
    """Analyze a message and return recommendation metadata only (HITL gate)."""
    preference = get_preference(request.conversation_id)
    context = extract_context(
        request.text,
        historical_preference=preference,
        use_llm=request.use_llm,
    )
    suggestion = recommend(context, request.text)

    message_id = save_message(
        conversation_id=request.conversation_id,
        role="user",
        modality=request.input_modality.value,
        content=request.text,
        metadata={
            "detected_context": context.as_dict(),
            "recommendation": suggestion.model_dump(mode="json"),
            "recommended_at_ms": int(time.time() * 1000),
            "input_modality": request.input_modality.value,
        },
    )
    return {
        "conversation_id": request.conversation_id,
        "message_id": message_id,
        "current_modality": request.input_modality.value,
        "detected_context": context.as_dict(),
        "recommended_modality": suggestion.recommended_modality.value,
        "reason": suggestion.reason,
        "confidence": suggestion.confidence,
        "scores": suggestion.scores,
        "requires_user_confirmation": True,
    }


def _answer_with_context(context_summary: str, user_text: str) -> tuple[str, bool]:
    try:
        reply = text_qwen.chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah DISIAI, asisten adaptif. Gunakan ringkasan konteks "
                        "berikut agar percakapan tidak terputus:\n" + context_summary
                    ),
                },
                {"role": "user", "content": user_text},
            ]
        )
        return reply, False
    except RuntimeError:
        return (
            "(Mode demo: Qwen belum tersedia) Konteks sebelumnya sudah diterima. "
            f"Pertanyaan Anda: {user_text}",
            True,
        )


def _dispatch(
    modality: Modality,
    *,
    context_summary: str,
    user_text: str,
) -> tuple[str, dict[str, Any]]:
    answer, fallback = _answer_with_context(context_summary, user_text)
    media: dict[str, Any] = {"fallback": fallback}

    if modality == Modality.VOICE:
        try:
            media.update(voice_whisper.synthesize_sync(answer))
        except RuntimeError as exc:
            media["voice_error"] = str(exc)
    elif modality == Modality.AVATAR:
        try:
            speech = voice_whisper.synthesize_sync(answer)
            media.update(speech)
        except RuntimeError as exc:
            media["voice_error"] = str(exc)
        media.update(avatar_sadtalker.generate(answer, audio_url=media.get("audio_url")))

    return answer, media


@app.post("/confirm")
def confirm(request: ConfirmRequest) -> dict[str, Any]:
    """Apply the user's accept/reject decision, keep context, and log evaluation."""
    message = get_message(request.message_id)
    if not message or message["conversation_id"] != request.conversation_id:
        raise HTTPException(status_code=404, detail="Pesan tidak ditemukan")

    metadata = message["metadata"]
    context = DetectedContext.model_validate(metadata["detected_context"])
    suggestion = Recommendation.model_validate(metadata["recommendation"])
    recommended_at = int(metadata.get("recommended_at_ms", time.time() * 1000))

    accepted = request.decision == "accept"
    target = (
        request.target_modality or suggestion.recommended_modality
        if accepted
        else Modality(message["metadata"].get("input_modality", "text"))
    )

    context_summary = build_context_summary(
        request.conversation_id, exclude_message_id=request.message_id
    )
    answer, media = _dispatch(target, context_summary=context_summary, user_text=message["content"])

    if accepted:
        record_preference(request.conversation_id, target.value)

    save_message(
        conversation_id=request.conversation_id,
        role="assistant",
        modality=target.value,
        content=answer,
        metadata={"media": media, "context_summary": context_summary},
    )

    latency_ms = int(max(0, time.time() * 1000 - recommended_at))
    evaluation_id = _log_with_connection(
        conversation_id=request.conversation_id,
        detected_context=context.as_dict(),
        recommended_modality=suggestion.recommended_modality.value,
        user_decision=request.decision,
        switching_latency_ms=latency_ms,
        target_modality=target.value,
        reason=suggestion.reason,
        confidence=suggestion.confidence,
    )

    return {
        "conversation_id": request.conversation_id,
        "decision": request.decision,
        "active_modality": target.value,
        "answer": answer,
        "media": media,
        "context_summary": context_summary,
        "evaluation_id": evaluation_id,
        "switching_latency_ms": latency_ms,
    }


def _log_with_connection(**kwargs: Any) -> int:
    from memory import get_connection

    with get_connection() as connection:
        return log_evaluation(connection, **kwargs)


@app.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    audio_bytes = await file.read()
    try:
        return voice_whisper.transcribe(audio_bytes, filename=file.filename or "audio.wav")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: str) -> dict[str, Any]:
    from memory import get_messages

    return {"messages": get_messages(conversation_id)}


@app.get("/evaluations")
def evaluations(conversation_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    return {"logs": _logs_with_connection(conversation_id=conversation_id, limit=limit)}


def _logs_with_connection(**kwargs: Any) -> list[dict[str, Any]]:
    from memory import get_connection

    with get_connection() as connection:
        return get_evaluation_logs(connection, **kwargs)
