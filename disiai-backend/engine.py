"""Deterministic rule-based modality recommendation engine."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Intent(str, Enum):
    TECHNICAL = "technical"
    QUICK_QUESTION = "quick_question"
    INTERPERSONAL = "interpersonal"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Situation(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    HANDS_FREE = "hands_free"


class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    AVATAR = "avatar"


class DetectedContext(BaseModel):
    intent: Intent
    urgency: Urgency
    situation: Situation
    historical_preference: Modality | None = None
    signals: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "urgency": self.urgency.value,
            "situation": self.situation.value,
            "historical_preference": (
                self.historical_preference.value if self.historical_preference else None
            ),
            "signals": self.signals,
        }


class Recommendation(BaseModel):
    recommended_modality: Modality
    reason: str
    confidence: float
    scores: dict[str, float]


def _initial_scores() -> dict[Modality, float]:
    return {modality: 0.0 for modality in Modality}


def recommend(context: DetectedContext, text: str = "") -> Recommendation:
    """Apply the SRS rules and return a recommendation (never auto-switch)."""
    scores = _initial_scores()
    reasons: dict[Modality, list[str]] = {modality: [] for modality in Modality}

    def add(modality: Modality, weight: float, label: str) -> None:
        scores[modality] += weight
        reasons[modality].append(label)

    # Text: analytical/technical/long tasks, or public space (privacy).
    if context.intent == Intent.TECHNICAL:
        add(Modality.TEXT, 0.65, "tugas bersifat teknis/analitis")
    if len(text) > 400:
        add(Modality.TEXT, 0.15, "pesan panjang cocok dibaca sebagai teks")
    if any(marker in text.lower() for marker in ("```", "def ", "function ", "debug", "kode")):
        add(Modality.TEXT, 0.20, "pesan mengandung kode/teknis")
    if context.situation == Situation.PUBLIC:
        add(Modality.TEXT, 0.55, "ruang publik membutuhkan privasi teks")

    # Voice: hands-free or urgent/quick answers.
    if context.situation == Situation.HANDS_FREE:
        add(Modality.VOICE, 0.75, "kondisi hands-free lebih aman dengan suara")
    if context.urgency == Urgency.HIGH:
        add(Modality.VOICE, 0.65, "urgensi tinggi membutuhkan jawaban cepat")
    if context.intent == Intent.QUICK_QUESTION:
        add(Modality.VOICE, 0.15, "pertanyaan singkat")
    if 0 < len(text) <= 80:
        add(Modality.VOICE, 0.10, "pesan singkat")

    # Avatar: interview / reflective consultation / virtual face-to-face.
    if context.intent == Intent.INTERPERSONAL:
        add(Modality.AVATAR, 0.75, "interaksi interpersonal/simulasi terdeteksi")
    if any(
        keyword in text.lower()
        for keyword in ("wawancara", "interview", "konsultasi", "refleksi", "curhat", "tatap muka")
    ):
        add(Modality.AVATAR, 0.90, "kebutuhan simulasi wawancara/konsultasi/refleksi")
    if context.situation == Situation.PRIVATE:
        add(Modality.AVATAR, 0.20, "konteks privat mendukung tatap muka virtual")

    # Historical preference is only a small tie-breaker, never an override.
    if context.historical_preference:
        add(
            context.historical_preference,
            0.12,
            f"preferensi historis pengguna: {context.historical_preference.value}",
        )

    order = [Modality.TEXT, Modality.VOICE, Modality.AVATAR]
    winner = max(order, key=lambda modality: (scores[modality], -order.index(modality)))
    total = sum(scores.values())
    confidence = round(scores[winner] / total, 2) if total > 0 else 0.5
    confidence = min(0.95, max(0.55, confidence)) if scores[winner] > 0 else 0.5

    detail = "; ".join(reasons[winner]) if reasons[winner] else "tidak ada sinyal kuat; default Text"
    reason = (
        f"Rekomendasi {winner.value.capitalize()}: {detail}. "
        + "Skor: "
        + ", ".join(f"{m.value}={scores[m]:.2f}" for m in order)
        + "."
    )

    return Recommendation(
        recommended_modality=winner,
        reason=reason,
        confidence=confidence,
        scores={m.value: round(scores[m], 2) for m in order},
    )
