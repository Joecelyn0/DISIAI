# DISIAI Backend

FastAPI middleware untuk adaptive multimodal routing (Text / Voice / Avatar).

## Struktur

```
disiai-backend/
├── main.py                  # FastAPI app + endpoints + orkestrasi HITL
├── config.py                # settings dari .env (pydantic-settings)
├── extractor.py             # ekstraksi intent, urgency, situation, preference
├── engine.py                # rule engine + confidence + alasan rekomendasi
├── memory.py                # SQLite: conversations, messages, preferences, handoff summary
├── logger.py                # tabel evaluation_log (keputusan + latensi)
├── handlers/
│   ├── text_qwen.py         # Qwen 2.5 via Ollama OpenAI-compatible /v1
│   ├── voice_whisper.py     # Faster-Whisper STT + Edge-TTS (id-ID-GadisNeural)
│   └── avatar_sadtalker.py  # SadTalker + fallback video mock
├── static/mock/avatar.mp4   # video fallback untuk pengembangan tanpa GPU
├── requirements.txt
└── .env.example
```

## Menjalankan

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Ollama belum berjalan? Jalankan `ollama serve` lalu `ollama pull qwen2.5:7b-instruct`.
Tanpa Ollama, ekstraksi memakai heuristik dan jawaban memakai mode demo.

## Endpoint utama

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | status Ollama, DB, dan video mock |
| POST | `/context/extract` | 4 parameter konteks implisit |
| POST | `/recommend` | rekomendasi + metadata (belum berpindah modalitas) |
| POST | `/confirm` | `accept`/`reject`, ringkasan konteks, dispatch handler, evaluation log |
| POST | `/voice/transcribe` | audio → transkrip via Faster-Whisper |
| GET | `/media/...` | audio/video hasil handler |
| GET | `/evaluations` | log evaluasi untuk analisis latensi |

Contoh alur:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/recommend -Method Post -ContentType application/json `
  -Body '{"conversation_id":"c1","text":"Sibuk menyetir, segera jawab singkat"}'
# lalu kirim message_id ke /confirm dengan decision "accept" atau "reject"
```

## Integrasi frontend

Next.js memanggil backend lewat proxy same-origin:

- `/api/disiai/recommend`
- `/api/disiai/confirm`
- `/api/disiai/health`
- `/api/disiai/media/[...path]`

Halaman demo: `http://localhost:3000/disiai`.
