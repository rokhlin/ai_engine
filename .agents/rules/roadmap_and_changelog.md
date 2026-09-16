---
name: Roadmap and Changelog Tracking Rule
description: Ensures ROADMAP.md and CHANGELOG.md are updated whenever any feature or functionality is implemented.
---

# Updating ROADMAP.md and CHANGELOG.md

Whenever you implement any new feature, program, or functionality in this project, you MUST perform the following synchronization steps:

## 1. Update `ROADMAP.md`
- Check [ROADMAP.md](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/ROADMAP.md).
- **If the feature already exists in the roadmap**: Mark its checkbox as completed (`- [x]`).
- **If the feature does NOT exist in the roadmap**: Add it under the most appropriate section/category in [ROADMAP.md](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/ROADMAP.md) and mark it as completed (`- [x]`).

## 2. Update `CHANGELOG.md`
- Locate [CHANGELOG.md](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/CHANGELOG.md).
- Under `## [Unreleased]`, add a descriptive bullet point under the appropriate category (`### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, or `### Security`). If the section header does not exist, create it.
- Keep descriptions concise, informative, and past-tense.
- Do not create a new version tag (e.g. `## [1.1.0]`) unless explicitly requested to cut a release.
