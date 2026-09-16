# Project Roadmap

## 🚀 Completed Features (v1.0.0)

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

---

## 📋 Development Plan & Planned Features

### Architecture & Core Infrastructure
- [ ] **Comprehensive Documentation**: Create complete architecture, API, and setup documentation.
- [x] **Feature Development Guidelines**: Create developer guidelines and standards for contributing new features.
- [x] **TypeScript Support**: Migrate/add TypeScript support across the frontend and interface definitions.
- [x] **Modular Architecture & Interface Structuring**: Refactor files, models, and interfaces into clean, structured layers.
- [ ] **Error handling & recovery**: Handle errors gracefully and provide options for recovery. define rules for: 
    - source not available, 
    - file not exist
    - file was deleted
    - Transcription error
    - face recognition error
    - LLM API error (Gemini, LM Studio)
    - image analysis error
    - file processing error
    - db error
    
- [ ] **Test Coverage**: Expand unit, integration, and E2E test suites with high coverage.
- [x] **Docker / Host Decoupling**: Separate application core dependencies from local host environment for robust Docker deployment.

### Metadata, Processing & AI Pipeline
- [ ] **Transcription Integration**: Fully embed audio transcription into the main media analysis pipeline.
- [x] **Media Tagging System**: Support custom and AI-generated tags for media assets.
    - [x] AI Tag Recognition with configurable format (`categorized`, `flat`, `prefixed`) and custom target criteria / vocabulary.
    - [x] Dedicated endpoint `POST /api/ai/analyze-with-tags` and sidecar/SQLite persistence.
- [ ] **Metadata Editing**: Add full support for viewing and editing metadata.
- [x] **Write Analysis to File Metadata**: Option to embed analysis results directly back into media files (EXIF/XMP/ID3) and sidecars.
- [x] **Direct Metadata & Tag Writing for HEIC and HEVC**:
    - [x] Add dedicated `.heic` and `.hevc` metadata tracking with loss-free sidecar and database persistence.
    - [x] Add automated test suite validating in-file tag embedding and reading for media formats.
- [ ] **In-Viewer Metadata Editor**: Direct inline metadata editing within the preview modal/window.
- [ ] **AI Analysis Agent**: Implement dedicated autonomous AI agent for deep file analysis and insights.
- [ ] **Face Optimization Service**: Background service for periodic face embedding optimization and clustering.
- [ ] **Advanced Duplicate Grouping**: Enhanced UI and pipeline for managing duplicate and burst photos.
- [ ] **Semantic Search**: Vector-based semantic search across media content, transcripts, and metadata.
- [x] **Worker Queue**: Configure a queue of media files for analysis. Local model can process multiple files simultaneously. Gemini API should also have a parallel processing capability. 

### UI / UX & Visual Features
- [ ] **Navigation Overhaul**: Redesign layout, sidebars, and workflow navigation.
- [x] **Internationalization (i18n)**: Multi-language interface and localization support.
- [x] **Theme System**: Dark, Light, and custom theme presets.
- [ ] **Calendar Timeline View**: View photos on a chronological calendar grid by capture date/time.
- [ ] **Album Management**: Create, curate, and share custom photo and video albums.
- [ ] **Family Tree**: Interactive genealogy / family relationship visualization connected to recognized faces.
- [ ] **Photo Stories**: Dynamic story creation and presentation from photo series.
- [ ] **AI Story Generation**: Automatic AI narrative and story generation based on photo context.
- [ ] **Memories & Flashbacks**: "On this day" and smart milestone memory reminders.

### Security, Access Control & Admin
- [ ] **Authentication & Route Guards**: User login, JWT sessions, and route protection.
- [ ] **User & Role Management (RBAC)**: Multi-user support with custom roles and permission levels.
- [ ] **Admin Dashboard**: System health monitoring, background worker controls, and administrative settings.
- [ ] **Secret / Vault Folder**: Encrypted private folder protected by password/PIN.
- [ ] **Vault Search Exclusion**: Strictly exclude hidden and private vault items from global searches and indexing.

### Clients & Platforms
- [ ] **Mobile Application**: Mobile app support (iOS / Android / PWA) with responsive synchronization.
