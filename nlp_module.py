from utility import *
from config import OLLAMA_CONFIG, OCR_CONFIG, PROCESSING_CONFIG, SYSTEM_PROMPT, CACHE_CONFIG
from tqdm import tqdm

def validate_processing_results(folder_path: Path) -> Dict[str, Any]:
    """
    Returns:
      {
        'validation': { 'context': True, ... },
        'missing_files': ['user_prompt.json'],
        'invalid_json': ['llm_response.json']
      }
    """
    validation: Dict[str, bool] = {}
    missing: List[str] = []
    invalid: List[str] = []
    expected_files: Dict[str, str] = {
        'context': 'llm_context.json',
        'prompt': 'user_prompt.json',
        'response': 'llm_response.json',
        'error': 'llm_error.json',
        'extraction': 'extracted_reinsurance_data.json'
    }

    for file_type, filename in expected_files.items():
        file_path = folder_path / filename
        if not file_path.exists():
            validation[file_type] = False
            missing.append(filename)
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            validation[file_type] = True
        except json.JSONDecodeError:
            validation[file_type] = False
            invalid.append(filename)

    return {'validation': validation, 'missing_files': missing, 'invalid_json': invalid}

class PromptAndObtain:
    """Set system prompt, give context and obtain model response"""
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = model_name or OLLAMA_CONFIG["model_name"]

    def cleanup(self):
        """Cleanup resources - ADD THIS METHOD"""
        try:
            # Ollama library doesn't hold persistent connections
            # but we can clear any cached data
            gc.collect()
            self.logger = get_logger(__name__)
            self.logger.debug(f"Cleaned up resources for model {self.model_name}")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during cleanup: {e}")

    def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 4000,
            temperature: float = 0.1,
            top_k: int = 40,
            top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate response from local Ollama model using ollama library

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt/instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dictionary with response data and metadata
        """
        start_time: float = time.time()

        try:
            with tqdm(total=2, desc="Generating response", leave=False) as pbar:
                pbar.set_description("Sending request to model")

                result = ollama.generate(
                    model=self.model_name,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": top_p,
                        "top_k": top_k
                    }
                )
                pbar.update(1)

                pbar.set_description("Processing response")
                generation_time: float = time.time() - start_time
                response_text: str = result.get("response", "")
                pbar.update(1)

            return {
                "success": True,
                "response": response_text,
                "model": result.get("model", self.model_name),
                "generation_time": generation_time,
                "tokens_generated": len(response_text.split()),
                "metadata": {
                    "prompt_length": len(prompt),
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "max_tokens": max_tokens
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - start_time
            }

class ModelCapabilityDetector:
    """Detect model capabilities via 'ollama show' and tune parameters accordingly"""

    DEFAULT_MODEL_SIZE_B = 7.0  # default assume 7B if unknown
    MIN_MODEL_SIZE_B = 0.05  # 50M as 0.05B
    MAX_MODEL_SIZE_B = 2048.0  # 2T for sanity cap
    DEFAULT_CONTEXT_WINDOW = 2048

    def __init__(self):
        self.default_model_size_b = None
        self.logger = get_logger(__name__)
        self.cached_capabilities: Dict[str, Dict[str, Any]] = {}
        self.default_model_size_b = (
            self.default_model_size_b if self.default_model_size_b is not None else self.DEFAULT_MODEL_SIZE_B
        )

    def get_model_capabilities(self, language_model_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch model capabilities using ollama show command.

        Returns:
            Dict with keys: context_length, parameter_size, quantization, etc.
        """
        if language_model_name in self.cached_capabilities:
            return self.cached_capabilities[language_model_name]

        try:
            # ollama.show() returns model info
            info = ollama.show(language_model_name)
            quant = self._detect_quantization(info)
            parsed_params_b = self._parse_parameter_size(info, language_model_name)
            ctx = self._detect_context_window(info)

            # Extract relevant fields
            capabilities = {
                'model_name': language_model_name,
                'details': info.get('details', {}),
                'parameters': info.get('parameters', {}),
                'modelfile': info.get('modelfile', ''),
                'quantization': self._detect_quantization(info),
                'parameters_size': self._parse_parameter_size(info),
                'context_window': self._detect_context_window(info),
                'raw_info': info
            }

            self.cached_capabilities[language_model_name] = capabilities
            self.logger.info(f"Model {language_model_name} capabilities: {capabilities}")
            return capabilities

        except Exception as e:
            self.logger.error(f"Failed to get capabilities for {language_model_name}: {e}")
            return None

    def _detect_quantization(self, info: Dict) -> str:
        """
        Extract quantization level and normalize to canonical tokens:
        returns one of ('q4','q5','q8','int8','fp16','fp32','unknown')
        """
        # Search several fields where quantization might be reported
        candidates = []

        details = info.get('details') or {}
        # common fields
        for key in ('quantization', 'quantization_level', 'format', 'model_format', 'tags', 'description'):
            v = details.get(key) if isinstance(details, dict) else None
            if v:
                candidates.append(str(v))

        # top-level modelfile or model name metadata
        for key in ('modelfile', 'model', 'name'):
            v = info.get(key)
            if v:
                candidates.append(str(v))

        joined = ' '.join(candidates).lower()

        # look for patterns
        # q4, q4_0, q5, q8, int8, int4, 4-bit, 8-bit, fp16, bf16, fp32
        if re.search(r'\bq4\b|\bq4[_\-]?\d*\b|4bit|\b4-bit\b', joined):
            return 'q4'
        if re.search(r'\bq5\b|\bq5[_\-]?\d*\b', joined):
            return 'q5'
        if re.search(r'\bq8\b|\bq8[_\-]?\d*\b|8bit|\b8-bit\b', joined):
            return 'q8'
        if re.search(r'\bint8\b|\buint8\b', joined):
            return 'int8'
        if re.search(r'\bfp16\b|\bbf16\b|\bhalf\b', joined):
            return 'fp16'
        if re.search(r'\bfp32\b|\bsingle-?precision\b|\bfloat32\b', joined):
            return 'fp32'

        # fallback: if text contains 'quant' and a number of bits
        m = re.search(r'(\d+)\s*-?\s*bit', joined)
        if m:
            bits = int(m.group(1))
            if bits <= 4:
                return 'q4'
            if bits <= 8:
                return 'q8'
            if bits <= 16:
                return 'fp16'

        return 'unknown'

    def _parse_parameter_size(self, info: Dict, model_name: str = '') -> float:
        """
        Parse parameter size and return in billions (float). Uses multiple heuristics:
        - details.parameter_size if provided (may be number or string like '7B')
        - info.parameters if it contains numeric count
        - model/modelfile strings: look for '7B', '13B', '125M', raw integer bytes
        - fallback to configured default.
        """
        details = info.get('details') or {}
        candidate = details.get('parameter_size') or details.get('params') or None

        # If provided as numeric (raw param count)
        if isinstance(candidate, (int, float)):
            return self._normalize_params_to_b(candidate)

        if isinstance(candidate, str):
            parsed = self._parse_param_string(candidate)
            if parsed is not None:
                return parsed

        # Try top-level 'parameters' field
        top_params = info.get('parameters')
        if isinstance(top_params, (int, float)):
            return self._normalize_params_to_b(top_params)
        if isinstance(top_params, str):
            parsed = self._parse_param_string(top_params)
            if parsed is not None:
                return parsed

        # Try modelfile and model_name
        for text in (info.get('modelfile', ''), info.get('model', ''), model_name):
            if not text:
                continue
            parsed = self._parse_param_string(str(text))
            if parsed is not None:
                return parsed

        # Last-ditch: look for explicit large integer anywhere in details or raw_info
        raw = str(info.get('raw_info') or info)
        m = re.search(r'(\d{7,12})', raw)  # 7-12 digits ~ millions to trillions
        if m:
            try:
                val = int(m.group(1))
                return self._normalize_params_to_b(val)
            except Exception:
                pass

        # Fallback to configured default
        self.logger.warning(
            f"Could not parse parameter size from info for model '{model_name}', using default {self.default_model_size_b}B")
        return float(self.default_model_size_b)

    def _normalize_params_to_b(self, numeric_params: float) -> float:
        """
        Accepts either raw parameter count (e.g., 7000000000) or already billions (7.0)
        and returns float in billions.
        """
        if numeric_params <= 100:  # probably already in billions (e.g., 7, 13)
            return float(numeric_params)
        # otherwise treat as raw count
        return float(numeric_params) / 1e9

    def _parse_param_string(self, s: str) -> Optional[float]:
        """
        Parse strings like:
          - '7B', '7b', '13B', '1.3B', '700M', '770M'
          - '700,000,000', '7000000000'
          - 'parameters: 7000000000'
        returns float (billions) or None
        """
        if not s:
            return None

        s_clean = s.strip().replace(',', '').replace(' ', '').lower()

        # direct integer
        if s_clean.isdigit():
            try:
                return self._normalize_params_to_b(int(s_clean))
            except Exception:
                pass

        # match <number><suffix>
        m = re.match(r'([0-9]*\.?[0-9]+)([kmb])\b', s_clean)
        if m:
            val = float(m.group(1))
            suf = m.group(2)
            if suf == 'k':
                return val / 1_000_000  # thousand -> billions
            if suf == 'm':
                return val / 1_000  # million -> billions
            if suf == 'b':
                return val  # already in billions

        # match number followed by B/b or M/m explicitly (e.g., '1.3B', '770M')
        m2 = re.search(r'([0-9]*\.?[0-9]+)\s*([bBmM])', s)
        if m2:
            num = float(m2.group(1))
            unit = m2.group(2).lower()
            if unit == 'b':
                return num
            if unit == 'm':
                return num / 1000.0
        # nothing matched
        return None

    def _detect_context_window(self, info: Dict) -> int:
        """Extract context window size. Accepts '8192', '8k', '32k', etc."""
        details = info.get('details') or {}
        candidates = []

        for key in (
                'context length', 'context_window', 'context-window', 'context window', 'max context', 'max_context'):
            v = details.get(key)
            if v:
                candidates.append(str(v))

        # try top-level
        for key in ('context', 'context_length'):
            v = info.get(key)
            if v:
                candidates.append(str(v))

        joined = ' '.join(candidates).lower()

        # direct digits
        m = re.search(r'(\d{3,6})', joined)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

        # look for '8k', '32k', etc.
        m2 = re.search(r'(\d+)\s*[kK]\b', joined)
        if m2:
            try:
                return int(m2.group(1)) * 1000
            except Exception:
                pass

        # fallback default
        self.logger.debug("Context window not found in info; using default %d", self.DEFAULT_CONTEXT_WINDOW)
        return int(self.DEFAULT_CONTEXT_WINDOW)

    def tune_parameters(
            self,
            model: str,
            use_case: str = 'extraction'  # 'extraction', 'validation', 'quick'
    ) -> Dict[str, Any]:
        """
        Recommend tuned parameters based on model capabilities.

        Args:
            model: Name of model
            use_case: 'extraction' (detailed), 'validation' (fast), 'quick' (minimal)

        Returns:
            Dict with recommended: temperature, top_k, top_p, max_tokens
        """
        capabilities = self.get_model_capabilities(model)

        if not capabilities:
            self.logger.warning(f"Using default parameters for {model}")
            return self._default_parameters(use_case)

        param_size = capabilities.get('parameters_size')
        quantization = capabilities.get('quantization', 'unknown')
        context_window = capabilities.get('context_window', self.DEFAULT_CONTEXT_WINDOW)

        # Validate param_size and coerce to float in billions
        try:
            if param_size is None:
                raise ValueError("None")
            param_size = float(param_size)
            # clamp reasonable range
            if not (self.MIN_MODEL_SIZE_B <= param_size <= self.MAX_MODEL_SIZE_B):
                self.logger.debug(
                    f"Parameter size {param_size}B outside expected range; using default {self.default_model_size_b}B")
                param_size = float(self.default_model_size_b)
        except Exception:
            self.logger.warning(
                f"Could not interpret parameter size '{param_size}', using default {self.default_model_size_b}B")
            param_size = float(self.default_model_size_b)

        # Build recommendations
        params = {
            'model_name': model,
            'use_case': use_case,
            'model_size_b': param_size,
            'quantization': quantization,
            'context_window': context_window
        }

        # Temperature: lower for extraction (consistency), higher for diversity
        if use_case == 'extraction':
            params['temperature'] = 0.1  # Deterministic
        elif use_case == 'validation':
            params['temperature'] = 0.05  # Very deterministic
        else:  # quick
            params['temperature'] = 0.1

        # Top-K: more aggressive for smaller models
        if param_size and param_size <= 2:
            params['top_k'] = 20
        elif param_size and param_size <= 7:
            params['top_k'] = 30
        else:
            params['top_k'] = 40

        # Top-P: nucleus sampling
        params['top_p'] = 0.9

        # Max tokens: scale by context and use case
        if use_case == 'extraction':
            # For detailed extraction
            max_toks = int(context_window * 0.3)  # 30% of context
            max_toks = min(max_toks, 4000)
            max_toks = max(max_toks, 2000)
        elif use_case == 'validation':
            # Validation typically needs less output
            max_toks = int(context_window * 0.2)
            max_toks = min(max_toks, 2000)
            max_toks = max(max_toks, 1000)
        else:  # quick
            max_toks = int(context_window * 0.15)
            max_toks = min(max_toks, 1500)
            max_toks = max(max_toks, 500)

        params['max_tokens'] = max_toks

        # Quantization adjustments
        if quantization in ('q4', 'int4', '4bit'):
            params['temperature'] = min(params['temperature'] + 0.05, 0.2)
            params['top_k'] = max(params['top_k'] - 10, 5)
        elif quantization in ('int8', 'q8'):
            # small conservative tweak
            params['temperature'] = min(params['temperature'] + 0.02, 0.15)

        self.logger.info(f"Tuned parameters for {model}: {params}")

        return params

    def _default_parameters(self, use_case: str) -> Dict[str, Any]:
        """Fallback default parameters"""
        if use_case == 'extraction':
            return {
                'temperature': 0.1,
                'top_k': 40,
                'top_p': 0.9,
                'max_tokens': 4000
            }
        elif use_case == 'validation':
            return {
                'temperature': 0.05,
                'top_k': 30,
                'top_p': 0.9,
                'max_tokens': 2000
            }
        else:  # quick
            return {
                'temperature': 0.1,
                'top_k': 20,
                'top_p': 0.9,
                'max_tokens': 1500
            }


class ModelChain:
    """
    Sequential multi-stage LLM processing with proper resource cleanup.

    For hardware-limited setups:
    1. Run primary model → save result → cleanup model
    2. Load validation model → validate → cleanup
    3. Merge results

    This prevents OOM by ensuring only one model is in memory at a time.
    """

    def __init__(
            self,
            primary_model: str,
            validation_model: Optional[str] = None,
            use_validation: bool = True,
            capability_detector: Optional['ModelCapabilityDetector'] = None
    ):
        self.primary_model = primary_model
        self.validation_model = validation_model or primary_model
        self.use_validation = use_validation
        self.logger = get_logger(__name__)
        self.detector = capability_detector or ModelCapabilityDetector()

    def extract_with_chain(
            self,
            prompt: str,
            context: Dict[str, Any],
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using model chain with sequential cleanup.

        Returns:
            Combined result with both stages
        """
        results = {
            'primary': None,
            'validation': None,
            'final': None,
            'confidence': 0.0,
            'stages_used': [],
            'timings': {}
        }

        # STAGE 1: Primary extraction
        self.logger.info(f"Stage 1: Primary extraction with {self.primary_model}")

        # Get tuned parameters for primary model
        primary_params = self.detector.tune_parameters(
            self.primary_model,
            use_case='extraction'
        )

        # Create temporary support instance for primary extraction
        primary_support = PromptAndObtain(self.primary_model)

        try:
            primary_start = time.time()
            primary_result = primary_support.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=primary_params['max_tokens'],
                temperature=primary_params['temperature'],
                # Pass tuned top_k if needed
            )
            primary_time = time.time() - primary_start
            results['timings']['primary_generation'] = primary_time
            results['primary'] = primary_result
            results['stages_used'].append('primary')

            self.logger.info(f"Primary extraction completed in {primary_time:.2f}s")

            if not primary_result.get('success'):
                self.logger.error("Primary extraction failed")
                results['final'] = None
                results['confidence'] = 0.0
                return results

            # Parse primary result
            try:
                primary_data = self._parse_json_response(primary_result['response'])
                results['primary']['parsed_data'] = primary_data
                self.logger.debug(f"Primary extraction parsed successfully")
            except Exception as e:
                self.logger.error(f"Primary parsing failed: {e}")
                results['final'] = None
                results['confidence'] = 0.0
                return results

        finally:
            # CRITICAL: Cleanup primary model before loading validation model
            primary_support.cleanup()
            del primary_support
            gc.collect()
            self.logger.info("Primary model cleaned up, memory freed")
            time.sleep(1)  # Give OS time to reclaim memory

        # STAGE 2: Sequential Validation (only if different model and enabled)
        if (self.use_validation and
                self.validation_model != self.primary_model and
                primary_data):

            self.logger.info(f"Stage 2: Validation with {self.validation_model}")

            # Get tuned parameters for validation model
            validation_params = self.detector.tune_parameters(
                self.validation_model,
                use_case='validation'
            )

            # Create temporary support instance for validation
            validation_support = PromptAndObtain(self.validation_model)

            try:
                # Create validation prompt
                validation_prompt = self._create_validation_prompt(
                    original_extraction=primary_data,
                    context=context
                )

                validation_start = time.time()
                validation_result = validation_support.generate_response(
                    prompt=validation_prompt,
                    system_prompt=system_prompt,
                    max_tokens=validation_params['max_tokens'],
                    temperature=validation_params['temperature']
                )
                validation_time = time.time() - validation_start
                results['timings']['validation_generation'] = validation_time
                results['validation'] = validation_result
                results['stages_used'].append('validation')

                self.logger.info(f"Validation completed in {validation_time:.2f}s")

                if validation_result.get('success'):
                    try:
                        validation_data = self._parse_json_response(
                            validation_result['response']
                        )
                        results['validation']['parsed_data'] = validation_data

                        # Merge results (validation takes precedence)
                        results['final'] = self._merge_extractions(
                            primary_data,
                            validation_data
                        )
                        results['confidence'] = 0.95  # High confidence with validation
                        self.logger.info("Validation successful, high confidence result")

                    except Exception as e:
                        self.logger.warning(f"Validation parsing failed: {e}")
                        results['final'] = primary_data
                        results['confidence'] = 0.7  # Fallback to primary
                else:
                    self.logger.warning("Validation generation failed")
                    results['final'] = primary_data
                    results['confidence'] = 0.7

            finally:
                # CRITICAL: Cleanup validation model
                validation_support.cleanup()
                del validation_support
                gc.collect()
                self.logger.info("Validation model cleaned up, memory freed")

        else:
            # No validation needed
            results['final'] = primary_data
            results['confidence'] = 0.8  # Medium confidence without validation
            self.logger.info("No validation stage - using primary extraction only")

        return results

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        cleaned = response.strip()

        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()
        return json.loads(cleaned)

    def _create_validation_prompt(
            self,
            original_extraction: Dict[str, Any],
            context: Dict[str, Any]
    ) -> str:
        """
        Create validation prompt that checks extracted values.
        Focuses on critical fields only to reduce token use.
        """
        critical_fields = {
            'insured': original_extraction.get('insured', 'TBD'),
            'cedant': original_extraction.get('cedant', 'TBD'),
            'total_sum_insured': original_extraction.get('total_sum_insured', 'TBD'),
            'currency': original_extraction.get('currency', 'TBD'),
            'period_of_insurance': original_extraction.get('period_of_insurance', 'TBD'),
            'premium': original_extraction.get('premium', 'TBD'),
            'reinsurance_type': original_extraction.get('reinsurance_type', 'TBD')
        }

        validation_prompt = f"""VALIDATION TASK: Review the extracted reinsurance data below.

            Original Extraction (Critical Fields):
            {json.dumps(critical_fields, indent=2)}

            Source Context (first 1500 chars):
            {str(context)[:1500]}

            VALIDATION RULES:
            1. Check if values match the source context
            2. Ensure currency matches TSI
            3. Validate dates are in correct format
            4. Flag any obvious errors or inconsistencies
            5. If a field looks wrong, provide correction

            OUTPUT: Return corrected JSON with same structure. Keep correct values unchanged. Use "TBD" only if truly missing.
            Return JSON only, no explanations.
        """
        return validation_prompt

    def _merge_extractions(
            self,
            primary: Dict[str, Any],
            validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge extractions: validation corrects, primary provides defaults.
        """
        merged = primary.copy()

        for key, val_value in validation.items():
            # Skip TBD from validation - keep primary's value
            if val_value == "TBD" or val_value is None:
                continue

            # Validation's value overrides primary
            if val_value != "TBD":
                merged[key] = val_value
                self.logger.debug(f"Updated {key}: {primary.get(key)} -> {val_value}")

        return merged

def extract_with_caching_and_chaining(
        support: PromptAndObtain,
        prompt: str,
        context: Dict[str, Any],
        cache_dir: Path,
        use_cache: bool = True,
        use_chain: bool = False,
        validation_model: Optional[str] = None,
        system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract with optional caching and chaining, using tuned parameters.

    Args:
        support: Support instance with primary model
        prompt: Extraction prompt
        context: Email context
        cache_dir: Cache directory
        use_cache: Enable prompt caching
        use_chain: Enable model chaining
        validation_model: Optional validation model
        system_prompt: Optional system prompt

    Returns:
        Dict with keys: 'extracted_data', 'success', 'confidence', 'timings', 'stages_used'
    """
    logger = get_logger(__name__)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup cache
    cache = PromptCache(cache_dir) if use_cache else None
    detector = ModelCapabilityDetector()

    # Check cache first
    if cache:
        cached = cache.get_cached_response(prompt, support.model_name)
        if cached:
            logger.info("Cache HIT: Using cached LLM response")
            return cached['response']

    logger.info("Cache MISS: Running extraction")

    # Use default system prompt if not provided
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    # Extract
    if use_chain and validation_model and validation_model != support.model_name:
        logger.info(f"Using model chain: {support.model_name} -> {validation_model}")

        chain = ModelChain(
            primary_model=support.model_name,
            validation_model=validation_model,
            use_validation=True,
            capability_detector=detector
        )
        result = chain.extract_with_chain(
            prompt,
            context,
            system_prompt=system_prompt
        )

        extraction = {
            'extracted_data': result['final'],
            'success': result['final'] is not None,
            'confidence': result['confidence'],
            'timings': result['timings'],
            'stages_used': result['stages_used'],
            'primary_result': result['primary'],
            'validation_result': result.get('validation')
        }

    else:
        # Single model extraction with tuned parameters
        logger.info(f"Single-model extraction: {support.model_name}")

        params = detector.tune_parameters(support.model_name, use_case='extraction')
        logger.debug(f"Tuned parameters: {params}")

        result = support.generate_response(
            prompt,
            system_prompt=system_prompt,
            max_tokens=params['max_tokens'],
            temperature=params['temperature']
        )

        if result.get('success'):
            # Parse JSON from response
            raw_response = result['response'].strip()

            # Clean markdown code blocks
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:]

            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]

            raw_response = raw_response.strip()

            try:
                extracted_data = json.loads(raw_response)
                logger.debug("JSON parsed successfully from response")
            except json.JSONDecodeError as e:
                logger.warning(f"Direct JSON parse failed: {e}, trying regex extraction")
                # Try regex extraction as fallback
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group())
                        logger.info("Extracted JSON via regex fallback")
                    except json.JSONDecodeError:
                        logger.error(f"Regex extraction also failed")
                        extracted_data = None
                else:
                    logger.error("No JSON object found in response")
                    extracted_data = None

            extraction = {
                'extracted_data': extracted_data,
                'success': extracted_data is not None,
                'confidence': 0.8,  # Single model confidence
                'timings': {
                    'primary_generation': result.get('generation_time', 0)
                },
                'stages_used': ['primary'],
                'raw_response': raw_response,
                'model_used': result.get('model')
            }
        else:
            logger.error(f"Generation failed: {result.get('error')}")
            extraction = {
                'extracted_data': None,
                'success': False,
                'confidence': 0.0,
                'timings': {},
                'stages_used': [],
                'error': result.get('error')
            }

    # Cache result if successful and caching enabled
    if cache and extraction.get('success') and extraction.get('extracted_data'):
        cache.cache_response(prompt, extraction, support.model_name)
        logger.info("Cached extraction result")

    return extraction

def validate_extraction_json(data: Dict[str, Any]) -> ReinsuranceExtraction:
    """
    Validate and parse extraction JSON from LLM.

    Args:
        data: Raw dictionary from LLM response

    Returns:
        Validated ReinsuranceExtraction object

    Raises:
        ValidationError: If data doesn't match schema
    """
    return ReinsuranceExtraction(**data)

