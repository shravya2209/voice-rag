FROM python:3.11-slim

WORKDIR /app

# Memory & Thread optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_TRIM_THRESHOLD_=100000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    LOG_LEVEL=INFO

# System deps for faiss & compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids downloading 2GB CUDA binaries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code & pre-computed index
COPY . .

# Ensure data and indexes directories exist
RUN mkdir -p data indexes evaluation

# Expose Render port
EXPOSE 10000

# Start server with single worker and lazy-loaded memory model
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
