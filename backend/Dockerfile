# Use Python 3.12.4 slim as base
FROM python:3.12.4-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install wheel for faster builds
RUN pip install --upgrade pip wheel

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with caching
# This layer will be cached unless requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]