# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model/ ./model/
COPY api/ ./api/
COPY ui/ ./ui/
COPY train.py .

# Create necessary directories
RUN mkdir -p data/dataset data/generated checkpoints logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
