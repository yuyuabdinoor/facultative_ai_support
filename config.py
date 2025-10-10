"""
Configuration file for the OCR + LLM Pipeline
"""

# Local Ollama Configuration
OLLAMA_CONFIG = {
    "model_name": "gemma3:270m",
    # mistrallite:latest llama3.2:1b or phi4-mini-reasoning:3.8b or phi4-reasoning:14b or qwen3:8b or qwen3:4b or gemma3:270m or gemma3:4b-it-qat
    "timeout": 300,  # 5 minutes
    "max_tokens": 3000,
    "temperature": 0.1
}

# OCR Configuration
OCR_CONFIG = {
    "device": "cpu",  # or "gpu" or "gpu:0,1" for multiple GPUs
    "cpu_threads": 2,  # Number of CPU threads for OCR, dependent on compute available
    "enable_mkldnn": True,  # Enable MKL-DNN acceleration on CPU
    "det_limit_side_len": 2880,  # Max side length for detection (higher = better quality but slower)
    "text_det_thresh": 0.3,  # Detection threshold for text pixels
    "text_det_box_thresh": 0.6,  # Threshold for text region boxes
    "text_recognition_batch_size": 4,  # Batch size for recognition, dependent on compute available
    "confidence_threshold": 0.7,  # Minimum confidence for including OCR text
}

# Processing Configuration
PROCESSING_CONFIG = {
    "save_visuals": True,  # Save annotated images and OCR results
    "pdf_dpi": 300,  # DPI for PDF rendering (200-300 recommended)
    "max_text_length": 80000,  # Maximum text length before truncation
    "confidence_threshold": 0.7,
    "truncate_attachments": True,
    "max_attachment_length": 15000,  # Maximum attachment text length dependent on llm context window
    "skip_processed": True,  # Skip folders with existing results
    "header_detection": 0.6,  # Header row detection threshold
    "data_quality": 0.1,  # Minimum data density for sheets
}

# Caching Configuration
CACHE_CONFIG = {
    "enable_ocr_cache": True,
    "enable_llm_cache": True,
    "enable_prompt_cache": True,
    "prompt_similarity_threshold": 0.95,
    "enable_model_chaining": False,  # Set to True to use
    "validation_model": "phi4-reasoning:14b"  # For chaining
}

# Model Paths (optional - PaddleOCR will download if not found)
MODEL_PATHS = {
    "detection model": "PP-OCRv5_mobile_det_infer",
    "recognition model": "PP-OCRv5_mobile_rec_infer",
    "detection folder": 'PP-OCRv5_mobile_det_infer',
    "recognition folder": 'PP-OCRv5_mobile_rec_infer'
}

# File Type Support
SUPPORTED_EXTENSIONS = {
    "office": [".docx", ".pptx", ".xlsx", ".csv"],
    "images": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
    "documents": [".pdf"],
    "text": [".txt", ".log"],
}

# Output Configuration
OUTPUT_CONFIG = {
    "save_context": True,
    "save_master_prompt": True,
    "save_llm_result": True,
    "save_clean_extraction": True,
    "save_errors": True
}
