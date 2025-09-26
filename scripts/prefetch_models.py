#!/usr/bin/env python3
import os
import pathlib


def ensure_dirs():
    project_root = pathlib.Path(__file__).resolve().parents[1]
    cache_root = project_root / "backend" / ".cache"
    hf_cache = cache_root / "huggingface"
    st_cache = cache_root / "sentence-transformers"
    hf_cache.mkdir(parents=True, exist_ok=True)
    st_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["SENTENCE_TRANSFORMERS_CACHE"] = str(st_cache)
    return hf_cache, st_cache


def prefetch():
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        BartForConditionalGeneration,
        DebertaV2ForSequenceClassification,
        pipeline,
    )
    from sentence_transformers import SentenceTransformer

    hf_cache = os.environ.get("HF_HOME")

    # Zero-shot classifier
    pipeline("zero-shot-classification", model="facebook/bart-large-mnli", cache_dir=hf_cache)

    # NER general + FinBERT NER + Sentiment
    pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        cache_dir=hf_cache,
    )
    pipeline("ner", model="ProsusAI/finbert", aggregation_strategy="simple", cache_dir=hf_cache)
    pipeline("sentiment-analysis", model="ProsusAI/finbert", cache_dir=hf_cache)

    # Decision engine models
    AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir=hf_cache)
    AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir=hf_cache)
    BartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir=hf_cache)
    AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir=hf_cache)
    DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", cache_dir=hf_cache)
    AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", cache_dir=hf_cache)

    # Sentence embeddings
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_CACHE", ""))


def main():
    try:
        ensure_dirs()
        prefetch()
        print("Model prefetch complete.")
    except Exception as e:
        print(f"Model prefetch failed: {e}")
        raise


if __name__ == "__main__":
    main()


