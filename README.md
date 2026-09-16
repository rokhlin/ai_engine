# Media Cataloger (AI Engine)

An automated AI-powered media archive cataloging suite, facial recognition registry, and remote execution engine designed to run headless or as part of a decoupled **two-tier architecture**.

---

## 🏛 Multi-Machine Architecture

```
+-------------------------------------------------------+                 REST / HTTP (JSON)                +-------------------------------------------------------+
|                 media_cataloger_web                   | ------------------------------------------------> |                    media_cataloger                    |
|           Frontend & Media Library Service            |              CATALOGER_API_URL=:8001              |                Media Cataloger AI Service             |
|                                                       | <------------------------------------------------ |                                                       |
|  * React 19 SPA (Gallery, Family Tree, Face Tagging)  |               Status, Logs, Triggers              |  * Python 3.10 + OpenCV + FFmpeg Engine               |
|  * NestJS High-Speed REST Server (:8000)              |                                                   |  * InsightFace + Gemini / Local LLM / Whisper AI      |
|  * Local or Shared Media Storage & SQLite Database    |                                                   |  * FastAPI Remote Control & Pipeline Worker (:8001)   |
|  * Repository: github.com/rokhlin/media_cataloger_web |                                                   |  * Repository: github.com/rokhlin/media_cataloger     |
|  * Docker Image: media_cataloger_web                  |                                                   |  * Docker Image: media_cataloger                      |
+-------------------------------------------------------+                                                   +-------------------------------------------------------+
```

### 🔗 Repositories:
- **`media_cataloger` (This Repo)**: AI engine, facial recognition, Gemini/Local LLM multimodal description, audio transcription, SQLite database catalog, and FastAPI remote daemon (Machine B).
- **[`media_cataloger_web`](https://github.com/rokhlin/media_cataloger_web)**: Web UI dashboard, Interactive Family Tree Explorer, Media Viewer, Face tagging UI, and NestJS server (Machine A).

---

## ✨ Features (AI Engine)

- **AI Scene & Object Description**: Uses Google Gemini 2.5/3.0 or Local Vision LLMs (via LM Studio / Ollama) to analyze photos and videos, producing bilingual (EN/RU) semantic metadata.
- **Deep Facial Recognition**: InsightFace embedding extraction (512-d), cosine similarity clustering, unknown face registry, and persistent person linkage.
- **Audio Transcription**: OpenAI Whisper integration for extracting speech from video recordings.
- **Sidecar JSON Generation & SQLite Indexing**: Produces portable JSON metadata sidecars and indexes everything into an optimized SQLite WAL database.
- **FastAPI Remote Control**: Exposes REST endpoints on port `8001` for remote triggering of catalog scans, single-file processing, pause/resume/stop control, live progress updates, and face management.

---

## 🚀 Quick Deployment with Docker

### Standalone AI Service (Machine B):
```bash
# 1. Clone repository
git clone https://github.com/rokhlin/media_cataloger.git
cd media_cataloger

# 2. Configure .env with your Gemini API key or local model settings
cp data/config/.env.example data/config/.env

# 3. Launch Cataloger AI container on port 8001
docker compose up -d --build
```

> [!TIP]
> **Windows Docker Users**: See the comprehensive step-by-step Windows deployment guide: [DOCKER_WINDOWS_SETUP.md](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/DOCKER_WINDOWS_SETUP.md).


---

## 🛠 Local Development & CLI (`manage.py`)

The project includes a CLI runner `manage.py` (with wrappers `run.ps1` for PowerShell, `run.sh` for Bash, and `Makefile` for `make`):

| Command | Description | Example |
| :--- | :--- | :--- |
| `api` / `cataloger` | Run FastAPI remote control daemon on port `8001` | `python manage.py api` |
| `scan` | Run media cataloging pipeline locally via CLI | `python manage.py scan --force` |
| `test` | Run automated test suite (`pytest`) | `python manage.py test` |
| `db:status` | Display database status, applied migrations, and row counts | `python manage.py db:status` |
| `db:backup` | Create atomic hot backup of SQLite database | `python manage.py db:backup` |
| `db:migrate` | Apply pending schema migrations safely | `python manage.py db:migrate` |
| `up` | Start Cataloger Docker container in background | `python manage.py up` |
| `down` | Stop Docker containers | `python manage.py down` |
| `build` | Build Cataloger Docker image locally | `python manage.py build` |
| `info` | Display active environment configuration | `python manage.py info` |

> 📖 **Database Schema Evolution**: See [DATABASE_MIGRATIONS.md](DATABASE_MIGRATIONS.md) for details on schema migrations and zero-downtime database upgrades.

---

## ⚙️ Environment Configuration

Configuration is loaded from `data/config/.env` or `.env`:

```env
# Server Host and Port
API_PORT=8001
API_HOST=0.0.0.0

# AI Model Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
MODEL_PROVIDER=gemini # or "local"

# Local LLM (LM Studio / Ollama)
LM_HOST=http://localhost
LM_PORT=1234
LOCAL_API_BASE=http://localhost:1234/v1
LOCAL_MODEL_NAME=qwen2.5-vl-7b-instruct
FALLBACK_TO_LOCAL=true

# Whisper Audio Transcription
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Storage Paths
INPUT_FOLDERS=./media_input
OUTPUT_FOLDER=./media_output
DB_PATH=./media_output/catalog_history.db
```

---

## 📦 Docker Container Images

Multi-architecture images (`linux/amd64`, `linux/arm64`) are automatically published to GitHub Container Registry:
- **AI Cataloger Engine**: `ghcr.io/rokhlin/media_cataloger:latest`
