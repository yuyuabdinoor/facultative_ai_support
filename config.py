"""
Configuration file for the OCR + LLM Pipeline
"""

# Local Ollama Configuration
OLLAMA_CONFIG = {
    "model_name": "gemma3:270m",  # llama3.2:1b or phi4-mini-reasoning:3.8b or phi4-reasoning:14b or qwen3:8b or qwen3:4b or gemma3:270m
    "timeout": 300,  # 5 minutes
    "max_tokens": 2500,
    "temperature": 0.1
}

# OCR Configuration
OCR_CONFIG = {
    "device": "cpu",  # Change to "gpu" if you have GPU support
    "cpu_threads": 4,
    "enable_mkldnn": True,
    "det_limit_side_len": 1920,
    "text_det_thresh": 0.3,
    "text_det_box_thresh": 0.6,
    "text_recognition_batch_size": 8
}


# Processing Configuration
PROCESSING_CONFIG = {
    "save_visuals": True,
    "max_text_length": 80000,
    "confidence_threshold": 0.7,
    "truncate_attachments": True,
    "max_attachment_length": 15000
}

# Output Configuration
OUTPUT_CONFIG = {
    "save_context": True,
    "save_master_prompt": True,
    "save_llm_result": True,
    "save_clean_extraction": True,
    "save_errors": True
}
