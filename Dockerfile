# Builder stage for Python dependencies
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS python-builder
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and build essentials for compiling native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for multi-stage dependency isolation
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu

# Final runtime stage with NVIDIA CUDA & cuDNN libraries
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    API_PORT=8001 \
    API_HOST=0.0.0.0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ORT_LOG_LEVEL=3 \
    ORT_LOGGING_LEVEL=3

WORKDIR /app

# Install runtime system dependencies (Python3, ffmpeg for audio/video, OpenGL for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv packages from builder
COPY --from=python-builder /opt/venv /opt/venv

# Copy backend application source code
COPY . .

# Ensure volume mount target directories exist
RUN mkdir -p /app/data/config /app/media_input /app/media_output

EXPOSE 8001

# Run Media Cataloger API on port 8001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]

