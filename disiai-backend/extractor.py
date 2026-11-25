"""Implicit context extraction: intent, urgency, situation, preference.

Uses Qwen 2.5 when Ollama is reachable and falls back to deterministic
keyword heuristics so the middleware still works during early development.
"""

import re
from typing import Any

from config import settings
from engine import DetectedContext, Intent, Modality, Situation, Urgency
from handlers import text_qwen

TECHNICAL_PATTERNS = (
    r"analisis|analisa|teknis|kode|coding|debug|algoritma|database|query|"
    r"arsitektur|sistem|laporan|dokumen|panjang|bandingkan|evaluasi|optimasi"
)
INTERPERSONAL_PATTERNS = (
    r"wawancara|interview|konsultasi|refleksi|reflektif|curhat|tatap muka|"
    r"simulasi|peran|role.?play|terapi|bimbingan"
)
URGENT_PATTERNS = r"segera|mendesak|buruan|cepat|urgent|asap|harus sekarang|langsung saja"
HANDS_FREE_PATTERNS = (
    r"hands?.?free|sibuk mengetik|menyetir|nyetir|masak|olahraga|sedang jalan|"
    r"tangan penuh|tidak bisa mengetik|bukan di depan laptop"
)
PUBLIC_PATTERNS = r"publik|tempat umum|ruang publik|di kantor|di depan orang|orang lain|privasi"
PRIVATE_PATTERNS = r"privat|pribadi|sendiri|ruang sendiri|tidak ada orang"


def _matches(pattern: str, text: str) -> list[str]:
    return sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))


def _heuristic(text: str) -> dict[str, Any]:
    lowered = text.lower()
    signals: list[str] = []

    interpersonal = _matches(INTERPERSONAL_PATTERNS, lowered)
    technical = _matches(TECHNICAL_PATTERNS, lowered)
    urgent = _matches(URGENT_PATTERNS, lowered)
    hands_free = _matches(HANDS_FREE_PATTERNS, lowered)
    public = _matches(PUBLIC_PATTERNS, lowered)
    private = _matches(PRIVATE_PATTERNS, lowered)

    if interpersonal:
        intent = Intent.INTERPERSONAL
        signals.extend(f"interpersonal:{word}" for word in interpersonal)
    elif technical:
        intent = Intent.TECHNICAL
        signals.extend(f"teknis:{word}" for word in technical)
    elif "?" in text and len(text) <= 160:
        intent = Intent.QUICK_QUESTION
        signals.append("pertanyaan singkat")
    else:
        intent = Intent.QUICK_QUESTION if len(text) <= 80 else Intent.TECHNICAL
        signals.append("default dari panjang pesan")

    if urgent:
        urgency = Urgency.HIGH
        signals.extend(f"urgent:{word}" for word in urgent)
    elif any(word in lowered for word in ("nanti", "sebentar", "kalau sempat")):
        urgency = Urgency.MEDIUM
        signals.append("urgensi sedang")
    else:
        urgency = Urgency.LOW

    if hands_free:
        situation = Situation.HANDS_FREE
        signals.extend(f"hands_free:{word}" for word in hands_free)
    elif public:
        situation = Situation.PUBLIC
        signals.extend(f"publik:{word}" for word in public)
    elif private:
        situation = Situation.PRIVATE
        signals.extend(f"privat:{word}" for word in private)
    else:
        situation = Situation.PRIVATE

    return {
        "intent": intent,
        "urgency": urgency,
        "situation": situation,
        "signals": signals,
    }


def _llm_extract(text: str, *, use_llm: bool) -> dict[str, Any] | None:
    if not use_llm or not text_qwen.is_available(timeout=1.0):
        return None

    prompt = (
        "Ekstrak konteks dari pesan pengguna berikut.\n"
        "intent: technical | quick_question | interpersonal\n"
        "urgency: low | medium | high\n"
        "situation: private | public | hands_free\n"
        "Balas JSON: {\"intent\": ..., \"urgency\": ..., \"situation\": ..., "
        "\"signals\": [alasan singkat]}\n\n"
        f"Pesan: {text}"
    )
    try:
        result = text_qwen.generate_json(prompt)
        intent = Intent(result["intent"])
        urgency = Urgency(result["urgency"])
        situation = Situation(result["situation"])
        signals = result.get("signals", [])
        if not isinstance(signals, list):
            signals = []
        return {
            "intent": intent,
            "urgency": urgency,
            "situation": situation,
            "signals": [f"llm:{signal}" for signal in signals if isinstance(signal, str)]
            or ["llm"],
        }
    except (KeyError, ValueError, RuntimeError, TypeError):
        return None


def extract_context(
    text: str,
    *,
    historical_preference: Modality | None = None,
    use_llm: bool | None = None,
) -> DetectedContext:
    """Extract the four SRS context parameters from a text/transcript message."""
    enable_llm = settings.extractor_use_llm if use_llm is None else use_llm
    detected = _llm_extract(text, use_llm=enable_llm) or _heuristic(text)

    return DetectedContext(
        intent=detected["intent"],
        urgency=detected["urgency"],
        situation=detected["situation"],
        historical_preference=historical_preference,
        signals=detected["signals"],
    )
