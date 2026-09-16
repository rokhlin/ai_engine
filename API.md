# Media Cataloger API Documentation

REST API documentation for remote management and monitoring of the **Media Cataloger** system.

The API is built using the **FastAPI** framework and provides a glassmorphic Web Control Dashboard, interactive OpenAPI documentation (Swagger / ReDoc), and endpoints to trigger synchronization, manage dynamic folder settings, and inspect or rename face clusters.

---

## 1. Overview & General Info

- **Base URL**: `http://<host>:<port>` (default: `http://localhost:8001`)
- **Interactive Swagger UI**: `http://localhost:8001/docs`
- **ReDoc Documentation**: `http://localhost:8001/redoc`

### Environment Variables
| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_HOST` | `0.0.0.0` | Host address for the Uvicorn web server |
| `API_PORT` | `8001` | Port number for the Uvicorn web server |

---

## 2. Pipeline Execution Endpoints

### 2.1. Trigger Full Synchronization
Initiates a background pipeline run to scan configured input folders, process media files, and generate sidecar JSON metadata.

- **URL**: `/api/run`
- **Method**: `POST`
- **Query Parameters**:
  - `force` (*boolean*, optional, default: `false`): If `true`, forces reprocessing of all files, ignoring previous database history.

**Example Request**:
```bash
# Standard incremental sync (new / modified files only)
curl -X POST "http://localhost:8000/api/run"

# Force reprocess all files
curl -X POST "http://localhost:8000/api/run?force=true"
```

**Response (200 OK)**:
```json
{
  "status": "started",
  "message": "Cataloging pipeline started in the background."
}
```

**Error Responses**:
- `400 Bad Request`: A cataloging process is already in progress (`"A cataloging process is already running."`).

---

### 2.2. Analyze a Single File
Triggers an immediate background analysis for an individual photo or video file.

- **URL**: `/api/analyze-file`
- **Method**: `POST`
- **Query Parameters**:
  - `file` (*string*, required): Filename or relative path to the media file to analyze.

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/analyze-file?file=IMG_20240101.jpg"
```

**Response (200 OK)**:
```json
{
  "status": "started",
  "message": "Single file analysis for 'IMG_20240101.jpg' started in the background."
}
```

**Error Responses**:
- `400 Bad Request`: Empty filename or a cataloging process is already active.

---

### 2.3. Pause Execution
Pauses an active synchronization execution in a thread-safe manner.

- **URL**: `/api/pause`
- **Method**: `POST`

**Response (200 OK)**:
```json
{
  "status": "paused",
  "message": "Cataloging execution paused."
}
```

---

### 2.4. Resume Execution
Resumes a previously paused execution.

- **URL**: `/api/resume`
- **Method**: `POST`

**Response (200 OK)**:
```json
{
  "status": "resumed",
  "message": "Cataloging execution resumed."
}
```

---

### 2.5. Stop / Cancel Execution
Immediately requests cancellation of an active or paused pipeline execution.

- **URL**: `/api/stop`
- **Method**: `POST`

**Response (200 OK)**:
```json
{
  "status": "stopping",
  "message": "Stopping cataloging process..."
}
```

---

### 2.6. Get Execution Status
Retrieves the real-time execution status and progress of the background task worker.

- **URL**: `/api/status`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/status"
```

**Response (200 OK)**:
```json
{
  "status": "running",
  "current_task": "sync",
  "started_at": "2026-08-23T11:00:00.000000",
  "finished_at": null,
  "error": null,
  "target_file": null,
  "progress": {
    "current": 15,
    "total": 120,
    "percent": 12,
    "current_file": "photo.jpg",
    "stage": "Processing photo 15/120"
  }
}
```

**Possible `status` values**:
- `idle`: No active task; system is idle.
- `running`: A sync or single file analysis task is currently executing.
- `paused`: The running task is paused by the user.
- `stopped`: The task was cancelled/stopped by the user.
- `completed`: The most recent task completed successfully.
- `failed`: The task failed due to an error (see `error` field for details).

---

### 2.7. Get Execution Logs
Returns the last 200 lines from the run log file (`cataloger_run.log`).

- **URL**: `/api/logs`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/logs"
```

**Response (200 OK)**:
```json
{
  "logs": "[START] Cataloging Pipeline sync initiated at 2026-08-22 11:00:00\nScanning folder...\n..."
}
```

---

## 3. Directory Configuration Endpoints

### 3.1. Get Current Directory Settings
Returns the active input and output folder paths, default values from `.env`, and indicators of whether custom persistent settings are active.

- **URL**: `/api/settings`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/settings"
```

**Response (200 OK)**:
```json
{
  "input_folders": ["/app/media_input", "/nas/photos"],
  "output_folder": "/app/media_output",
  "default_input_folders": ["/app/media_input"],
  "default_output_folder": "/app/media_output",
  "is_custom_input": true,
  "is_custom_output": false
}
```

---

### 3.2. Update Directory Settings
Updates the active input folders and output folder dynamically without requiring a service restart. Changes are persisted in `settings.json`.

- **URL**: `/api/settings`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "input_folders": [
    "/data/media_archive/2024",
    "/data/media_archive/2025"
  ],
  "output_folder": "/data/catalog_output"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/settings" \
     -H "Content-Type: application/json" \
     -d '{"input_folders": ["/data/media_archive/2024"], "output_folder": "/data/catalog_output"}'
```

**Response (200 OK)**:
```json
{
  "status": "success",
  "message": "Settings updated successfully.",
  "input_folders": ["/data/media_archive/2024"],
  "output_folder": "/data/catalog_output"
}
```

**Error Responses**:
- `400 Bad Request`: Cannot update folder configuration while a sync or analysis is in progress.
- `500 Internal Server Error`: Failed to persist settings to disk.

---

### 3.3. Open Native Folder Picker
Opens the native OS directory selection dialog window (available when running natively in desktop environments).

- **URL**: `/api/select-folder`
- **Method**: `POST`

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/select-folder"
```

**Response (200 OK)**:
```json
{
  "folder": "C:\\Users\\User\\Pictures"
}
```

---

## 4. Face Registry Endpoints

### 4.1. List Registered Faces
Returns all registered faces from the database face registry.

- **URL**: `/api/faces`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/faces"
```

**Response (200 OK)**:
```json
[
  {
    "face_id": "FACE_ID_01",
    "name": "Alice",
    "embedding_shape": [512]
  },
  {
    "face_id": "FACE_ID_02",
    "name": "Bob",
    "embedding_shape": [512]
  }
]
```

---

### 4.2. List Known Persons (Multi-Reference)
Returns unique known persons along with their full collection of reference face sample thumbnails covering time changes.

- **URL**: `/api/faces/persons`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/faces/persons"
```

**Response (200 OK)**:
```json
[
  {
    "person_id": "Alice",
    "name": "Alice",
    "reference_count": 2,
    "primary_image": "facess/FACE_ID_01.jpg",
    "reference_faces": [
      {
        "face_id": "FACE_ID_01",
        "image_path": "facess/FACE_ID_01.jpg",
        "confidence": 0.95,
        "source_file": "media/photo1.jpg",
        "created_at": "2026-08-23 10:00:00",
        "has_embedding": true
      },
      {
        "face_id": "FACE_ID_02",
        "image_path": "facess/FACE_ID_02.jpg",
        "confidence": 0.88,
        "source_file": "media/photo2.jpg",
        "created_at": "2026-08-23 10:05:00",
        "has_embedding": true
      }
    ]
  }
]
```

---

### 4.3. List Unrecognized Faces
Returns face crops saved in `OUTPUT_FOLDER/facess/` that were not recognized or had match/detection confidence < 70%.

- **URL**: `/api/faces/unrecognized`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/faces/unrecognized"
```

**Response (200 OK)**:
```json
[
  {
    "face_id": "FACE_ID_03",
    "name": "FACE_ID_03",
    "image_path": "facess/FACE_ID_03.jpg",
    "confidence": 0.62,
    "source_file": "media/group_pic.jpg",
    "created_at": "2026-08-23 10:10:00",
    "has_embedding": true
  }
]
```

---

### 4.4. Get Face Thumbnail Crop
Serves a cropped face image thumbnail from `OUTPUT_FOLDER/facess/`.

- **URL**: `/api/faces/image/{filename}`
- **Method**: `GET`

**Example Request**:
```bash
curl -X GET "http://localhost:8000/api/faces/image/FACE_ID_01.jpg" --output face.jpg
```

---

### 4.5. Assign an Unrecognized Face
Assigns an unrecognized face crop to an existing or new person name, making it an active reference face for that person.

- **URL**: `/api/faces/assign`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "face_id": "FACE_ID_03",
  "name": "Alice"
}
```

---

### 4.6. Rename a Face / Person
Renames a person across all their reference face entries.

- **URL**: `/api/faces/rename`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "face_id": "FACE_ID_01",
  "name": "Alice Smith"
}
```

---

### 4.7. List Unrecognized Face Similarity Groups
Returns unrecognized faces clustered by embedding cosine similarity into groups with occurrence counts and source file tracking.

- **URL**: `/api/faces/unrecognized-groups` (or `/api/faces/groups`)
- **Method**: `GET`
- **Query Parameters**: `threshold` (optional float, defaults to `0.70`)

**Response (200 OK)**:
```json
[
  {
    "group_id": "FACE_ID_03",
    "representative_face_id": "FACE_ID_03",
    "name": "FACE_ID_03",
    "primary_image": "facess/FACE_ID_03.jpg",
    "face_ids": ["FACE_ID_03", "FACE_ID_04"],
    "count": 2,
    "faces": [
      {
        "face_id": "FACE_ID_03",
        "name": "FACE_ID_03",
        "image_path": "facess/FACE_ID_03.jpg",
        "confidence": 0.92,
        "source_file": "media/photo1.jpg"
      }
    ],
    "source_files": ["media/photo1.jpg", "media/photo2.jpg"],
    "avg_confidence": 0.91
  }
]
```

---

### 4.8. Assign Group of Similar Faces
Assigns an entire cluster/group of face IDs to a person name in one batch, marking them as active reference faces and syncing sidecars.

- **URL**: `/api/faces/assign-group`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "face_ids": ["FACE_ID_03", "FACE_ID_04"],
  "name": "Bob"
}
```

---

### 4.9. Reset a Face Assignment
Resets a face assignment back to an unassigned candidate face (`is_reference = 0`), updating the face registry and sidecars.

- **URL**: `/api/faces/reset`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "face_id": "FACE_ID_03"
}
```

---

### 4.10. Reset Face Assignments by Filename
Resets all face assignments for faces originating from a specific filename or path.

- **URL**: `/api/faces/reset-by-filename` (or `/api/faces/reset-by-file`)
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "filename": "photo1.jpg"
}
```

**Response (200 OK)**:
```json
{
  "status": "success",
  "message": "Reset 2 face assignment(s) for file 'photo1.jpg'.",
  "reset_count": 2,
  "face_ids": ["FACE_ID_03", "FACE_ID_04"]
}
```

---

### 4.11. Delete a Face Entry
Deletes a face entry from registry and removes its crop image from `facess/`.

- **URL**: `/api/faces/delete`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body (JSON)**:
```json
{
  "face_id": "FACE_ID_03"
}
```

**Error Responses**:
- `404 Not Found`: The specified `face_id` was not found in the face registry.

