FROM python:3.10-slim

WORKDIR /app

# Install system deps needed for OCR and image rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      poppler-utils \
      tesseract-ocr \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment configuration
ENV PORT=8080
# Optional: set common tesseract data path (tweak if your image OCR requires different path)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
# App-level knobs (read these from code to enforce rate limits / max iterations)
ENV AGENT_MAX_ITERATIONS=10
ENV RATE_LIMIT_PER_MINUTE=60

EXPOSE 8080

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]