FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Hugging Face uses port 7860
EXPOSE 7860

ENV PORT=7860

CMD ["gunicorn", "wsgi:app", "--workers", "2", "--timeout", "300", "--bind", "0.0.0.0:7860"]
