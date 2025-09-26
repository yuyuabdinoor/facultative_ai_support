"""
Security middleware for FastAPI application
"""
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import time
import logging
from typing import Dict, Set
import re

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses
    """
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy - relax for docs endpoints
        if request.url.path in ["/docs", "/redoc"]:
            # More permissive CSP for API documentation
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "connect-src 'self' https://cdn.jsdelivr.net; "
                "frame-ancestors 'none'"
            )
        else:
            # Strict CSP for other endpoints
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
        response.headers["Content-Security-Policy"] = csp
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Initialize or clean old requests for this IP
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove requests older than 1 minute
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.requests[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxy headers"""
        # Check for forwarded headers (when behind a proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests for audit and monitoring
    """
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"from {request.client.host}"
        )
        
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.3f}s"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for basic input validation and sanitization
    """
    
    # Patterns for common attacks
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
    ]
    
    def __init__(self, app):
        super().__init__(app)
        self.sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip validation for certain endpoints
        if self._should_skip_validation(request):
            return await call_next(request)
        
        # Validate query parameters
        for key, value in request.query_params.items():
            if self._contains_malicious_content(value):
                logger.warning(f"Malicious content detected in query param {key}: {value}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected"
                )
        
        # Validate path parameters
        path = str(request.url.path)
        if self._contains_malicious_content(path):
            logger.warning(f"Malicious content detected in path: {path}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path"
            )
        
        return await call_next(request)
    
    def _should_skip_validation(self, request: Request) -> bool:
        """Skip validation for certain endpoints like file uploads"""
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/health"]
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    def _contains_malicious_content(self, content: str) -> bool:
        """Check if content contains malicious patterns"""
        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if pattern.search(content):
                return True
        
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if pattern.search(content):
                return True
        
        return False

class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle maintenance mode
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.maintenance_mode = False
        self.maintenance_message = "System is under maintenance"
        self.allowed_paths = ["/health", "/admin/maintenance"]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self.maintenance_mode and not self._is_allowed_path(request.url.path):
            return Response(
                content=f'{{"message": "{self.maintenance_message}"}}',
                status_code=503,
                media_type="application/json",
                headers={"Retry-After": "3600"}
            )
        
        return await call_next(request)
    
    def _is_allowed_path(self, path: str) -> bool:
        """Check if path is allowed during maintenance"""
        return any(path.startswith(allowed) for allowed in self.allowed_paths)
    
    def enable_maintenance(self, message: str = None):
        """Enable maintenance mode"""
        self.maintenance_mode = True
        if message:
            self.maintenance_message = message
    
    def disable_maintenance(self):
        """Disable maintenance mode"""
        self.maintenance_mode = False

class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with security considerations
    """
    
    def __init__(self, app, allowed_origins: Set[str] = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or {"http://localhost:3000"}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = request.headers.get("Origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            if origin and origin in self.allowed_origins:
                return Response(
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Max-Age": "86400"
                    }
                )
            else:
                return Response(status_code=403)
        
        response = await call_next(request)
        
        # Add CORS headers for actual requests
        if origin and origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

# Global maintenance mode instance
maintenance_middleware = MaintenanceModeMiddleware(None)