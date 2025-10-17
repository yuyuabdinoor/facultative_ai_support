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
                                    output_dir=out_root.parent  # Pass parent (text_detected_and_recognized)
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


def validate_startup_config() -> bool:
    """
    Validate all configuration before pipeline start
    Returns True if valid, raises detailed exceptions if not
    """
    logger = get_logger(__name__)
    issues = []

    # 1. Validate Ollama model exists
    try:
        models = ollama.list()
        model_names = [m.model if hasattr(m, 'model') else m.name
                       for m in (models.models if hasattr(models, 'models') else [])]

        if OLLAMA_CONFIG['model_name'] not in model_names:
            issues.append(
                f"Primary model '{OLLAMA_CONFIG['model_name']}' not found in Ollama. "
                f"Available: {model_names[:5]}"
            )
    except Exception as e:
        issues.append(f"Cannot connect to Ollama: {e}")

    # 2. Validate validation model if chaining enabled
    if CACHE_CONFIG.get('enable_model_chaining'):
        val_model = CACHE_CONFIG.get('validation_model')
        if not val_model:
            issues.append("Model chaining enabled but no validation_model specified")
        elif val_model not in model_names:
            issues.append(f"Validation model '{val_model}' not found")

    # 3. Validate OCR model paths
    det_dir = Path(MODEL_PATHS['detection folder'])
    rec_dir = Path(MODEL_PATHS['recognition folder'])

    if not det_dir.exists():
        logger.warning(f"OCR detection model not found at {det_dir} - will download on first use")

    if not rec_dir.exists():
        logger.warning(f"OCR recognition model not found at {rec_dir} - will download on first use")

    # 4. Validate config values
    if PROCESSING_CONFIG['max_text_length'] < 1000:
        issues.append("max_text_length too small (< 1000)")

    if OLLAMA_CONFIG['max_tokens'] > 128000:
        issues.append("max_tokens unreasonably large (> 128k)")

    if OCR_CONFIG['confidence_threshold'] < 0 or OCR_CONFIG['confidence_threshold'] > 1:
        issues.append("confidence_threshold must be 0-1")

    # 5. Check system resources
    mem = psutil.virtual_memory()
    if mem.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB
        logger.warning(f"Low available memory: {mem.available / 1024 ** 3:.1f}GB")

    # Report
    if issues:
        logger.error("Configuration validation FAILED:")
        for issue in issues:
            logger.error(f"  ❌ {issue}")
        raise ValueError(f"Invalid configuration: {len(issues)} issues found")

    logger.info("✓ Configuration validated successfully")
    return True


class ProgressTracker:
    """Track and persist pipeline progress for resumption capability"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.progress_file = root_path / ".cache" / "pipeline_progress.json"
        self.state = self._load_state()
        self.logger = get_logger(__name__)

    def _load_state(self) -> Dict[str, Any]:
        """Load existing progress or create new"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load progress: {e}")

        return {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'processed_folders': [],
            'failed_folders': [],
            'skipped_folders': [],
            'total_folders': 0,
            'status': 'running'
        }

    def _save_state(self):
        """Atomic save of progress state"""
        self.state['last_updated'] = datetime.now().isoformat()

        temp_file = self.progress_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            temp_file.replace(self.progress_file)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def mark_processed(self, folder_name: str):
        """Mark folder as successfully processed"""
        if folder_name not in self.state['processed_folders']:
            self.state['processed_folders'].append(folder_name)
            self._save_state()

    def mark_failed(self, folder_name: str, error: str):
        """Mark folder as failed with error"""
        self.state['failed_folders'].append({
            'folder': folder_name,
            'error': error[:500],  # Truncate long errors
            'timestamp': datetime.now().isoformat()
        })
        self._save_state()

    def should_process(self, folder_name: str, skip_processed: bool = True) -> bool:
        """Check if folder should be processed"""
        if not skip_processed:
            return True

        return folder_name not in self.state['processed_folders']

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        return {
            'total_processed': len(self.state['processed_folders']),
            'total_failed': len(self.state['failed_folders']),
            'total_skipped': len(self.state['skipped_folders']),
            'success_rate': (
                    len(self.state['processed_folders']) /
                    max(self.state['total_folders'], 1) * 100
            ),
            'runtime': (
                    datetime.fromisoformat(self.state['last_updated']) -
                    datetime.fromisoformat(self.state['started_at'])
            ).total_seconds()
        }

    def finalize(self, status: str = 'completed'):
        """Mark pipeline as complete"""
        self.state['status'] = status
        self.state['completed_at'] = datetime.now().isoformat()
        self._save_state()


def run_pipeline(base_folder: str, m_name: str = None, skip_processed: bool = True):
    """
    Main pipeline orchestration with centralized path management and shared cache.

    Args:
        base_folder: Root directory containing email folders
        m_name: Model name (overrides config if provided)
        skip_processed: Skip folders already processed

    Raises:
        FileNotFoundError: If base_folder doesn't exist
        NotADirectoryError: If base_folder is not a directory
        OllamaModelNotLoadedException: If model not available
        OllamaConnectionException: If cannot connect to Ollama
    """

    # STEP 0: Validate configuration FIRST (fail-fast)
    # This checks Ollama connection, model availability, etc.
    try:
        validate_startup_config()
    except (OllamaModelNotLoadedException, OllamaConnectionException) as e:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

    # STEP 1: Resolve and validate root path
    root_path = Path(base_folder).resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    # STEP 2: Create centralized cache at root level
    root_cache_dir = root_path / ".cache"
    root_cache_dir.mkdir(exist_ok=True)

    shared_cache = ExtractionCache(root_cache_dir)
    progress = ProgressTracker(root_path)

    # STEP 3: Create timestamped log folder
    log_folder = get_log_folder(root_path)
    logger = setup_logging(log_folder, log_level=LogLevel.INFO)

    # Banner
    logger.info("=" * 80)
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Root path: {root_path}")
    logger.info(f"Cache dir: {root_cache_dir}")
    logger.info(f"Log folder: {log_folder}")
    logger.info(f"Model: {m_name or OLLAMA_CONFIG['model_name']}")
    logger.info(f"Skip processed: {skip_processed}")
    logger.info("=" * 80)

    # STEP 4: Health checks (lightweight - already validated Ollama)
    health_checker = HealthChecker()

    try:
        health_results = health_checker.run_all_checks(
            m_name or OLLAMA_CONFIG['model_name'],
            root_path
        )
    except Exception as e:
        logger.error(f"Health checks failed: {e}")
        logger.warning("Proceeding anyway - some checks may have passed")
        # Don't fail pipeline, just log warning

    # Check critical health results
    if 'filesystem' in health_results and not health_results['filesystem'].healthy:
        logger.error("Filesystem check failed - cannot proceed")
        logger.error(f"Reason: {health_results['filesystem'].message}")
        return  #Early return instead of continuing

    logger.info("✓ Health checks completed")

    # STEP 5: Initialize components ONCE
    try:
        # This validates model availability (fail-fast if not available)
        with PromptAndObtain(m_name) as ds:

            #Test generation before processing
            logger.info("Testing model generation...")
            if not ds.test_generation():
                logger.error("Model test generation failed - cannot proceed")
                return
            logger.info("✓ Model test successful")

            # Initialize other components
            rate_limiter = RateLimiter(max_operations=10, window_seconds=60)
            input_validator = InputValidator()
            patterns = ReinsurancePatterns()
            memory_monitor = MemoryMonitor()

            # STEP 6: Discover folders to process
            subfolders = sorted(
                [f for f in root_path.iterdir()
                 if f.is_dir() and f.name not in ("logs", ".cache", ".prompt_cache")],
                key=lambda x: x.name
            )

            if not subfolders:
                logger.warning(f"No email folders found in {root_path}")
                return

            logger.info(f"Found {len(subfolders)} folders to process")
            progress.state['total_folders'] = len(subfolders)

            # STEP 7: Process each folder
            for subfolder in tqdm(subfolders, desc="Processing emails", unit="email"):
                # Validate folder path
                valid, errors = input_validator.validate_path(subfolder, root_path)
                if not valid:
                    logger.warning(f"Skipping invalid folder {subfolder.name}: {errors}")
                    progress.mark_failed(subfolder.name, f"Invalid path: {errors}")
                    continue

                # Check if already processed
                if not progress.should_process(subfolder.name, skip_processed):
                    logger.info(f"Skipping {subfolder.name} - already processed")
                    progress.state['skipped_folders'].append(subfolder.name)
                    continue

                # Process folder
                try:
                    process_single_folder(
                        subfolder=subfolder,
                        ds=ds,
                        rate_limiter=rate_limiter,
                        patterns=patterns,
                        memory_monitor=memory_monitor,
                        skip_processed=skip_processed,
                        shared_cache=shared_cache
                    )
                    progress.mark_processed(subfolder.name)
                    logger.info(f"✓ Completed: {subfolder.name}")

                except OllamaException as e:
                    # Ollama-specific errors (timeout, memory, etc.)
                    logger.error(
                        f"Ollama error processing {subfolder.name}: {e.message}",
                        extra={'custom_fields': e.details}
                    )
                    progress.mark_failed(subfolder.name, f"{type(e).__name__}: {e.message}")

                    # Decide whether to continue or abort
                    if isinstance(e, (OllamaConnectionException, OllamaModelNotLoadedException)):
                        # Fatal errors - cannot continue
                        logger.error("Fatal Ollama error - aborting pipeline")
                        break
                    else:
                        # Transient errors - continue with next folder
                        continue

                except Exception as e:
                    # Unexpected errors
                    logger.error(
                        f"Unexpected error processing {subfolder.name}: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    progress.mark_failed(subfolder.name, str(e))
                    continue  # Don't crash entire pipeline

            # STEP 8: Finalize and report
            progress.finalize('completed')
            summary = progress.get_summary()

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETE")
            logger.info(f"Total folders: {summary['total_processed'] + summary['total_failed']}")
            logger.info(f"✓ Successful: {summary['total_processed']}")
            logger.info(f"✗ Failed: {summary['total_failed']}")
            logger.info(f"⊘ Skipped: {summary['total_skipped']}")
            logger.info(f"Success rate: {summary['success_rate']:.1f}%")
            logger.info(f"Runtime: {summary['runtime']:.1f}s")
            logger.info("=" * 80)

            # Export metrics
            metrics_path = log_folder / "metrics.json"
            export_metrics(metrics_path)
            logger.info(f"Metrics exported to: {metrics_path}")

    except OllamaModelNotLoadedException as e:
        # Model not available when creating PromptAndObtain
        logger.error(f"Model not available: {e.message}")
        logger.error(f"Available models: {e.available_models}")
        logger.error(f"Suggestion: {e.details.get('suggestion')}")
        sys.exit(1)

    except OllamaConnectionException as e:
        # Cannot connect to Ollama service
        logger.error(f"Cannot connect to Ollama: {e.message}")
        logger.error(f"Suggestion: {e.details.get('suggestion')}")
        sys.exit(1)

    except KeyboardInterrupt:
        # User cancelled
        logger.warning("Pipeline interrupted by user")
        progress.finalize('interrupted')
        summary = progress.get_summary()
        logger.info(f"Partial completion: {summary['total_processed']} folders processed")
        sys.exit(130)

    except Exception as e:
        # Unexpected fatal error
        logger.error(f"Fatal error in pipeline: {type(e).__name__}: {e}", exc_info=True)
        progress.finalize('failed')
        sys.exit(1)


root_folder = "thursday evening test data version two"

run_pipeline(
    base_folder=root_folder,
    m_name=OLLAMA_CONFIG['model_name'],
    skip_processed=PROCESSING_CONFIG['skip_processed']
)

