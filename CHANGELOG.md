# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **AI Tag Recognition & Structured Tagging Pipeline (`src/utils/gemini.py`, `src/utils/local_model.py`, `src/workers.py`, `api.py`)**:
  - Implemented configurable AI tag generation supporting three output formats: `categorized` (`category:tag`), `flat` (`tag`), and `prefixed` (`#tag`).
  - Added custom `target_tags` target criteria / vocabulary matching evaluated in a single LLM inference pass across both Gemini API and Local Model (LM Studio).
  - Extended Pydantic schemas `PhotoAnalysis` and `VideoAnalysis` with `TagItem` normalization and `content_type` classification (`documents`, `social`, `nature`, `animals`, `screenshots`, `family`, `non_family`, `events`, `travel`, `other`).
  - Added new REST endpoint `POST /api/ai/analyze-with-tags` in `api.py` and dynamic tag formatting options in `RuntimeSettings`, `RunSyncPayload`, and `AnalyzeFilePayload`.
  - Added structured tag persistence into sidecar JSON files and SQLite `media_tags` table.
- **Organize Files & Metadata Tagging Pipeline**:
  - Full support for library organization criteria (Year, Month, Common Event name, Content Type classification).
  - Multi-format media tagging with loss-free metadata persistence in sidecars and SQLite across image and video formats, including Apple HEIC (`.heic`) and HEVC/H.265 (`.hevc`, `.mp4`, `.mov`).
  - High-performance chunked tensor operations for GPU/CPU duplicate discovery and similarity threshold grouping.
- **AI Engine Security Integration & Python Client**:
  - Implemented `MediaCatalogerClient` (`media_cataloger_client.py` and `src/client.py`) with JWT session management, automatic token refresh on 401 Unauthorized, RBAC permission headers (`admin_panel`, `edit_metadata`, `manage_faces`, `vault_access`), and Secret Vault session unlocking.
  - Implemented `POST /api/auth/login`, `POST /api/vault/unlock`, `POST /api/vault/lock`, and `GET /api/vault/status` endpoints with HMAC-SHA256 token verification in `api.py`.
  - Added `POST /api/media/metadata` endpoint for bidirectional metadata ingestion (summaries, descriptions, localized attributes, tags, EXIF, transcripts, and timeline events) into database and disk sidecar JSON files.
  - Added Secret Vault isolation guarantee in `src/workers.py`: skips `.vault`, `vault`, `secret_vault` folders during scanning and cataloging unless an authorized vault session is provided.
  - Enhanced `POST /api/faces/assign` to seamlessly accept `person_name`, `file`, and `confidence` payloads as specified in `AI_ENGINE_SECURITY_INTEGRATION.md`.
  - Added automated test suite `tests/test_security_integration.py` and `tests/test_client.py` validating security endpoints and client workflows.
- **Frontend & Media Recognition Decoupling (Two Dedicated Applications)**:

  - **`Media Library` Application (`frontend/`)**: Converted into a standalone TypeScript application powered by **NestJS** and **React 19 (Vite)**. Houses all media library management, face registry operations, settings, and folder browsing. Exposes mobile-ready REST endpoints with OpenAPI/Swagger at `/api/docs`, utilizes `better-sqlite3` in WAL mode for sub-millisecond query performance on low-power hardware (such as Intel Celeron N3450 with 8GB RAM), and proxies AI pipeline triggers to `media_cataloger`.
  - **`media_cataloger` Business Logic & AI Service**: Pure Python FastAPI service dedicated to heavy media recognition, InsightFace embeddings, Whisper speech-to-text, ffmpeg video processing, and Gemini/Local LLM semantic analysis, exposing analysis endpoints on port 8001.
  - **Independent Standalone Deployments**: Packaged into two dedicated Dockerfiles (`frontend/Dockerfile` for Media Library and root `Dockerfile` for media_cataloger) capable of running on separate physical machines without coupled container dependencies.
- **Modular Frontend Architecture & Dedicated Models**: Refactored frontend interfaces, models, types, and enums into a modular structure under `src/models/` (`status.ts`, `media.ts`, `face.ts`, `settings.ts`, `modals.ts`, `i18n.ts`, `index.ts`). Split `TranslationDictionary` into dedicated per-component interfaces (`HeaderTranslations`, `ExecutionControlsTranslations`, `GalleryTranslations`, `LightboxTranslations`, `FaceRegistryTranslations`, `SettingsTranslations`, `LogsTranslations`, `CommonTranslations`), and extracted all hardcoded strings into `translations.ts`.
- **Internationalization (i18n) & Localized Photo Metadata**: Implemented bilingual (English 🇬🇧 / Russian 🇷🇺) support across the dashboard. Added a language switcher in the Header and Settings modal with localStorage persistence, full interface translations (`LanguageContext` and `translations.ts`), dynamic photo metadata localization (showing Russian/English AI descriptions, summaries, environment, lighting, weather, time of day, OCR, EXIF analysis, and video transcription), and a new backend endpoint `/api/media/sidecar` for loading complete sidecar metadata.
- **Frontend TypeScript Support**: Migrated entire frontend React codebase from JavaScript/JSX to TypeScript (`.tsx`/`.ts`), introduced comprehensive type definitions in `src/types/index.ts`, configured `tsconfig.json`, `vite.config.ts`, and added automated typechecking script (`npm run typecheck`).

## [1.0.0] - Initial Release

### Added
- **Photo Cataloging & Analysis**: EXIF extraction, GPS reverse geocoding, face recognition with local face registry matching, semantic analysis via Gemini API, and JSON sidecar file generation.
- **Video Content Analysis**: Video compression via ffmpeg, 1fps frame extraction, face presence timecode tracking, and semantic video analysis via Gemini File API.
- **Duplicate & Burst Shot Grouping**: Photo clustering (capture time <= 15s + pHash similarity), batch analysis via Gemini to assess differences, detect defects, and select the best frame.
- **Bilingual Search Support**: Populates English and Russian fields (`*_ru`) in sidecar JSON outputs.
- **Incremental Sync History**: SQLite database tracking to skip re-processing unchanged files.
- **Local Model Support**: Integration with LM Studio for local models, with fallback to local when Gemini limits are exhausted.
- **Whisper Integration**: Local speech-to-text with CPU fallback.
- **FastAPI Service**: Built-in REST API with OpenAPI/Swagger and ReDoc UI.
- **Web Dashboard**: Glassmorphic frontend dashboard with execution controls (run, pause, resume, stop full sync, analyze single file).
- **Task Runner (`manage.py`)**: Unified script to manage Docker containers, local dev servers, and tests.
- **Automated Testing & Verification**: `pytest` configuration and end-to-end `verify_run.py` script.
