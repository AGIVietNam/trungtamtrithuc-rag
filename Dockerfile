FROM python:3.11-slim

# ffmpeg cần thiết cho Whisper phiên âm video
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads data/logs

EXPOSE 8000

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
