# SafeEar backend (CPU).  Build: docker build -t safeear-backend .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAFEEAR_DATA_ROOT=/data/safeear

# libsndfile backs soundfile; ffmpeg lets librosa read compressed uploads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Vendor the ReDimNet source so the model loads without GitHub at runtime.
RUN python scripts/bootstrap_redimnet.py || echo "redimnet bootstrap skipped"

RUN useradd --create-home safeear && mkdir -p /data/safeear && chown -R safeear /app /data/safeear
USER safeear

EXPOSE 8000
# One worker: detection state lives in process memory (see README notes on scaling).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
