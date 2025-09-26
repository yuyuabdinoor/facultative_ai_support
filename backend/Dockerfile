# Use Python 3.12.4 slim as base
FROM python:3.12.4-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
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

# Install PaddleOCR (CPU)
RUN pip install --no-cache-dir paddlepaddle==2.6.1 paddleocr==2.7.0.3

# Set cache directories for models
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_CACHE=/app/.cache/sentence-transformers

# Prefetch Hugging Face and Sentence-Transformers models at build time
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers && \
    python - <<'PY'
import os
os.environ["HF_HOME"] = "/app/.cache/huggingface"
from transformers import (AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, BartForConditionalGeneration, DebertaV2ForSequenceClassification,
                          pipeline)
from sentence_transformers import SentenceTransformer

# Zero-shot
pipeline("zero-shot-classification", model="facebook/bart-large-mnli", cache_dir=os.environ["HF_HOME"])

# NER general + FinBERT NER + Sentiment
pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple", cache_dir=os.environ["HF_HOME"])
pipeline("ner", model="ProsusAI/finbert", aggregation_strategy="simple", cache_dir=os.environ["HF_HOME"])
pipeline("sentiment-analysis", model="ProsusAI/finbert", cache_dir=os.environ["HF_HOME"])

# Decision models
AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir=os.environ["HF_HOME"])
AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir=os.environ["HF_HOME"])
BartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir=os.environ["HF_HOME"])
AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir=os.environ["HF_HOME"])
DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", cache_dir=os.environ["HF_HOME"])
AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=os.environ["HF_HOME"])

# Sentence embeddings
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="/app/.cache/sentence-transformers")
print("Model prefetch complete")
PY

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]