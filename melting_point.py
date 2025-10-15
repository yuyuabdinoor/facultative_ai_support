from mail_info_extraction_module import *
from nlp_module import *
from utility import *


def validate_config(config_dict: Dict[str, Any]) -> AppConfig:
    """
    Validate application configuration.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Validated AppConfig object

    Raises:
        ValidationError: If config is invalid
    """
    return AppConfig.from_dict(config_dict)


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use as filesystem directory name.

    Disallow: < > : " / \ | ? *
    This function replaces them with underscores.
    """
    invalid_chars = {
        ':': '_',
        '/': '_',
        '\\': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        '?': '_',
        '*': '_',
    }

    sanitized = model_name
    for char, replacement in invalid_chars.items():
        sanitized = sanitized.replace(char, replacement)

    return sanitized


def process_attachments_with_cache(
        processor: EmailProcessor,
        email_data: Dict[str, Any],
        ds: DetectAndRecognize,
        cache: ExtractionCache,
        timestamp: str,
        logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Process attachments with intelligent caching.
    Skip OCR/extraction if already cached.
    """
    attachment_texts: List[Dict[str, Any]] = []
    attachments: List[Dict[str, Any]] = email_data.get('saved_attachments', [])

    # Create output root ONCE for new extractions
    out_root: Optional[Path] = None
    if PROCESSING_CONFIG['save_visuals']:
        out_root = Path(processor.subfolder_path) / "text_detected_and_recognized" / timestamp
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"OCR output directory: {out_root}")

    # Process each attachment
    for attachment in tqdm(attachments, desc="Processing attachments", unit="file", leave=False):
        file_path: str = attachment['path']
        file_path_obj = Path(file_path)

        if file_path_obj.suffix.lower() in ['.json']:
            continue

        # Check cache first
        if cache.has_ocr_cache(file_path_obj):
            logger.info(f"Using cached extraction for {attachment['filename']}")
            cached_data = cache.get_cached_ocr_data(file_path_obj)
            if cached_data:
                attachment_texts.append({
                    'file': attachment['filename'],
                    'text': cached_data.get('text', ''),
                    'method': cached_data.get('method', 'cached'),
                    'cached': True,
                    'time': 0.0
                })
                continue

    # Process only uncached attachments in ONE call to processor
    uncached = [a for a in attachments
                if not cache.has_ocr_cache(Path(a['path']))]

    if uncached:
        new_texts = processor.process_attachments_with_ocr(
            email_data,
            ds,
            save_visuals=PROCESSING_CONFIG['save_visuals']
        )
        attachment_texts.extend(new_texts if new_texts else [])

        # Register newly processed in cache
        for new_text in (new_texts or []):
            if not new_text.get('cached'):
                # Find corresponding file
                for att in attachments:
                    if att['filename'] == new_text['file']:
                        file_path_obj = Path(att['path'])

                        # out_root is where visuals were saved
                        if out_root:
                            try:
                                cache.register_ocr_cache(
                                    file_path=file_path_obj,
                                    method=new_text.get('method', 'unknown'),
                                    timestamp=timestamp,
                                    output_dir=out_root  # Pass parent (text_detected_and_recognized)
                                )
                                logger.debug(f"Registered cache for {att['filename']}")
                            except Exception as e:
                                logger.warning(f"Failed to register cache: {e}")
                        break

    return attachment_texts


def save_processing_results_with_cache(
        model_dir: Path,
        context: Dict[str, Any],
        llm_prompt: str,
        llm_result: Dict[str, Any],
        timestamp: str
) -> None:
    """
    Save processing results with timestamp in model-specific directory.
    """
    ts_iso = datetime.now().isoformat()

    with tqdm(total=4, desc="Saving results", unit="file", leave=False) as pbar:

        # Save LLM context (shared across runs)
        pbar.set_description("Saving LLM context")
        context_path: Path = model_dir / 'llm_context.json'
        context_data: Dict[str, Any] = {
            "context": context,
            "timestamp": ts_iso,
            "metadata": {
                "email_subject": context.get('email', {}).get('subject', ''),
                "total_attachments": context.get('total_attachments', 0),
                "processing_times": context.get('processing_times', {})
            }
        }
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        pbar.update(1)

        pbar.set_description("Saving user prompt")
        prompt_path: Path = model_dir / 'user_prompt.json'
        prompt_data: Dict[str, Any] = {
            "user_prompt": llm_prompt,
            "timestamp": ts_iso,
            "prompt_length": len(llm_prompt),
        }
        with open(prompt_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
        pbar.update(1)

        # Save timestamped LLM response
        if llm_result.get("success", False):
            pbar.set_description("Saving LLM response")
            result_path: Path = model_dir / f'llm_response_{timestamp}.json'

            metadata: Dict[str, Any] = llm_result.get("metadata", {})
            result_data: Dict[str, Any] = {
                "success": True,
                "extracted_data": llm_result.get("extracted_data", {}),
                "raw_response": llm_result.get("raw_response", ""),
                "timing": {
                    "generation_time_seconds": llm_result.get("generation_time", 0.0),
                    "tokens_generated": llm_result.get("tokens_generated", 0),
                    "timestamp": ts_iso
                },
                "model_info": {
                    "model_used": llm_result.get("model_used", "unknown"),
                    "temperature": metadata.get("temperature", 0.1),
                    "max_tokens": metadata.get("max_tokens", 3000)
                }
            }

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            pbar.update(1)

            # Save clean extraction
            pbar.set_description("Saving clean extraction")
            clean_path: Path = model_dir / f'extracted_reinsurance_data_{timestamp}.json'
            with open(clean_path, 'w', encoding='utf-8') as f:
                json.dump(llm_result.get("extracted_data", {}), f, indent=2, ensure_ascii=False)
            pbar.update(1)
        else:
            error_path: Path = model_dir / f'llm_error_{timestamp}.json'
            error_data: Dict[str, Any] = {
                "success": False,
                "error": llm_result.get("error", "Unknown error"),
                "timestamp": ts_iso
            }
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            pbar.update(2)


@track_performance('folder.process')
def process_single_folder(
        subfolder: Path,
        ds: PromptAndObtain,
        rate_limiter: RateLimiter,
        patterns: ReinsurancePatterns,
        memory_monitor: MemoryMonitor,
        shared_cache: ExtractionCache,
        skip_processed: bool = True,
):
    """Process a single email folder with intelligent caching"""
    folder_logger = create_logger_with_context(__name__, folder=subfolder.name)

    try:
        folder_logger.info(f"Processing folder: {subfolder.name}")

        cache = shared_cache
        m = ds.model_name
        model_cache_dir = cache.folder_path / sanitize_model_name(m)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Check if LLM extraction already exists for this model
        has_llm_cache, cached_response_path = cache.has_llm_cache(m)

        if has_llm_cache and skip_processed:
            folder_logger.info(f"Found cached LLM extraction for model {m}")
            with open(cached_response_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            folder_logger.info(f"Using cached extraction from {cached_response_path.name}")
            return  # EXIT EARLY - everything cached

        # Rate limiting
        if not rate_limiter.acquire():
            wait_time = rate_limiter.wait_time()
            folder_logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        # Memory check
        memory_monitor.check_and_cleanup()

        # Extract email
        processor = EmailProcessor(subfolder)
        email_data = processor.get_complete_data()

        if not email_data:
            folder_logger.error("No email data found")
            return

        folder_logger.info(f"Email: {email_data['metadata']['subject']}")

        # Process attachments with caching
        attachment_texts = process_attachments_with_cache(
            processor=processor,
            email_data=email_data,
            ds=ds,
            cache=cache,
            timestamp=timestamp,
            logger=folder_logger
        )

        folder_logger.info(f"Processed {len(attachment_texts)} attachments")

        # Create LLM context
        context, llm_prompt = create_llm_context(email_data, attachment_texts)

        # Generate context hash for cache validation
        context_hash = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()
        prompt_hash = hashlib.sha256(llm_prompt.encode()).hexdigest()

        # Extract candidates using patterns (optional enhancement)
        combined_text = llm_prompt
        candidates = extract_with_patterns(combined_text, patterns)

        if candidates:
            folder_logger.debug("Pattern extraction candidates", extra={'candidates': candidates})

        # Run LLM extraction with caching and optional chaining
        folder_logger.info("Running LLM extraction...")

        extraction = extract_with_caching_and_chaining(
            support=ds,
            prompt=llm_prompt,
            context=context,
            cache_dir=model_cache_dir,
            use_cache=CACHE_CONFIG.get('enable_prompt_cache', True),
            use_chain=CACHE_CONFIG.get('enable_model_chaining', False),
            validation_model=CACHE_CONFIG.get('validation_model')
            if CACHE_CONFIG.get('enable_model_chaining') else None,
            system_prompt=SYSTEM_PROMPT
        )

        # Build llm_result from extraction
        llm_result = {
            "success": extraction is not None and extraction.get('extracted_data') is not None,
            "extracted_data": extraction.get('extracted_data') if extraction else None,
            "model_used": m,
            "generation_time": extraction.get('timings', {}).get('primary_generation', 0)
            if isinstance(extraction, dict) else 0
        }

        # Validate extraction if successful
        if llm_result.get("success") and llm_result.get("extracted_data"):
            try:
                # Handle both dict and ReinsuranceExtraction objects
                extraction_data = llm_result["extracted_data"]
                if isinstance(extraction_data, dict):
                    validated_extraction = validate_extraction_json(extraction_data)
                else:
                    validated_extraction = extraction_data

                completeness = validated_extraction.get_completeness_score()
                missing_critical = validated_extraction.get_missing_critical_fields()

                folder_logger.info(
                    f"Extraction validated - Completeness: {completeness:.1f}%",
                    extra={
                        'completeness': completeness,
                        'missing_critical': missing_critical
                    }
                )

                # Store validated version
                if isinstance(validated_extraction, ReinsuranceExtraction):
                    llm_result["extracted_data"] = validated_extraction.model_dump()

                llm_result["completeness_score"] = completeness
                llm_result["missing_critical_fields"] = missing_critical

            except Exception as e:
                folder_logger.error("Extraction validation failed")
                log_exception(folder_logger, e, {'folder': subfolder.name})

        # Save results with timestamp
        save_processing_results_with_cache(
            model_dir=model_cache_dir,
            context=context,
            llm_prompt=llm_prompt,
            llm_result=llm_result,
            timestamp=timestamp
        )

        # Register in cache
        cache.register_llm_cache(
            model_name=m,
            timestamp=timestamp,
            context_hash=context_hash,
            prompt_hash=prompt_hash
        )

        folder_logger.info(f"Results saved to {model_cache_dir.name}/{timestamp}")

    except Exception as e:
        folder_logger.error(f"Error processing folder: {e}")
        log_exception(folder_logger, e, {'folder': subfolder.name})
        # traceback.print_exc()


def get_log_folder(root_folder: Path) -> Path:
    """Create timestamped log subfolder within root folder"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%A")  # Includes day name
    log_folder = Path(root_folder) / "logs" / timestamp
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder


def run_pipeline(base_folder: str, m_name: str = None, skip_processed: bool = True):
    """
    Main pipeline orchestration with centralized path management and shared cache
    """
    # STEP 1: Resolve and validate root path FIRST
    root_path = Path(base_folder)

    # Validate it exists and is a directory
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    # STEP 2: Create centralized cache ONCE at root level
    root_cache_dir = root_path / ".cache"
    root_cache_dir.mkdir(exist_ok=True)

    # NEW: Create shared cache instance
    shared_cache = ExtractionCache(root_cache_dir)

    # STEP 3: Create timestamped log folder
    log_folder = get_log_folder(root_path)
    logger = setup_logging(log_folder, log_level=LogLevel.INFO)

    logger.info("=" * 80)
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Root path: {root_path}")
    logger.info(f"Cache dir: {root_cache_dir}")
    logger.info(f"Log folder: {log_folder}")
    logger.info("=" * 80)

    # STEP 4: Health checks
    health_checker = HealthChecker()
    health_results = health_checker.run_all_checks(
        m_name or OLLAMA_CONFIG['model_name'],
        root_path
    )

    if not health_results.get('filesystem').healthy:
        logger.error("Filesystem check failed - cannot proceed")
        return

    logger.info("Health checks completed")

    # STEP 5: Initialize components ONCE
    with PromptAndObtain(m_name) as ds:
        rate_limiter = RateLimiter(max_operations=10, window_seconds=60)
        input_validator = InputValidator()
        patterns = ReinsurancePatterns()
        memory_monitor = MemoryMonitor()

        # STEP 6: Process folders with shared cache INSTANCE (not directory)
        subfolders = sorted(
            [f for f in root_path.iterdir()
             if f.is_dir() and f.name not in ("logs", ".cache", ".prompt_cache")],
            key=lambda x: x.name
        )

        logger.info(f"Found {len(subfolders)} folders to process")

        for subfolder in tqdm(subfolders, desc="Processing emails", unit="email"):
            valid, errors = input_validator.validate_path(subfolder, root_path)
            if not valid:
                logger.warning(f"Skipping invalid folder: {errors}")
                continue

            process_single_folder(
                subfolder=subfolder,
                ds=ds,
                rate_limiter=rate_limiter,
                patterns=patterns,
                memory_monitor=memory_monitor,
                skip_processed=skip_processed,
                shared_cache=shared_cache
            )

        logger.info("Pipeline complete")
        metrics_path = log_folder / "metrics.json"
        export_metrics(metrics_path)


root_folder = "thursday evening test data version two"

run_pipeline(
    base_folder=root_folder,
    m_name=OLLAMA_CONFIG['model_name'],
    skip_processed=PROCESSING_CONFIG['skip_processed']
)
