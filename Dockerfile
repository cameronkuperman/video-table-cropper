FROM python:3.12-slim

# Install FFmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create working directories
RUN mkdir -p uploads output frames

# Railway injects PORT at runtime
EXPOSE ${PORT:-8080}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 600
