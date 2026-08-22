FROM python:3.11-slim

WORKDIR /app

# System deps for faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data/index dirs
RUN mkdir -p data indexes evaluation

# Expose port (Render assigns $PORT dynamically)
EXPOSE 10000

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=10000
ENV LOG_LEVEL=INFO

# Start server — shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
