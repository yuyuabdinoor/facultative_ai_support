"""
Comprehensive API integration tests
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
import json
from datetime import datetime

from app.main import app
from app.core.auth import create_access_token

# Test client
client = TestClient(app)

# Test data
TEST_USER_DATA = {
    "username": "testuser",
    "password": "testpassword123",
    "email": "test@example.com",
    "full_name": "Test User"
}

ADMIN_USER_DATA = {
    "username": "admin",
    "password": "secret"
}

class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_login_success(self):
        """Test successful login"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "invalid",
                "password": "invalid"
            }
        )
        assert response.status_code == 401
    
    def test_get_current_user(self):
        """Test getting current user info"""
        # First login to get token
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        token = login_response.json()["access_token"]
        
        # Get user info
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == ADMIN_USER_DATA["username"]
    
    def test_refresh_token(self):
        """Test token refresh"""
        # First login to get tokens
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

class TestDocumentAPI:
    """Test document management endpoints"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers"""
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_document_upload(self, auth_headers):
        """Test document upload"""
        # Create a test file
        test_file_content = b"Test PDF content"
        files = {"file": ("test.pdf", test_file_content, "application/pdf")}
        
        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["filename"] == "test.pdf"
    
    def test_document_validation(self, auth_headers):
        """Test document validation"""
        test_file_content = b"Test content"
        files = {"file": ("test.txt", test_file_content, "text/plain")}
        
        response = client.post(
            "/api/v1/documents/validate",
            files=files,
            headers=auth_headers
        )
        # Should fail validation for unsupported file type
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] == False
    
    def test_list_documents(self, auth_headers):
        """Test listing documents"""
        response = client.get(
            "/api/v1/documents/",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_document_unauthorized(self):
        """Test document access without authentication"""
        response = client.get("/api/v1/documents/")
        assert response.status_code == 401

class TestApplicationAPI:
    """Test application management endpoints"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers"""
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_application(self, auth_headers):
        """Test creating an application"""
        application_data = {
            "status": "pending"
        }
        
        response = client.post(
            "/api/v1/applications/",
            json=application_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
    
    def test_list_applications(self, auth_headers):
        """Test listing applications"""
        response = client.get(
            "/api/v1/applications/",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "applications" in data
        assert "total" in data
        assert "page" in data
    
    def test_get_application_stats(self, auth_headers):
        """Test getting application statistics"""
        response = client.get(
            "/api/v1/applications/stats/overview",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_applications" in data
        assert "approval_rate" in data

class TestAnalyticsAPI:
    """Test analytics and reporting endpoints"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers"""
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_dashboard_metrics(self, auth_headers):
        """Test dashboard metrics"""
        response = client.get(
            "/api/v1/analytics/dashboard",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_applications" in data
        assert "approval_rate" in data
        assert "avg_processing_time_hours" in data
    
    def test_trend_analysis(self, auth_headers):
        """Test trend analysis"""
        response = client.get(
            "/api/v1/analytics/trends/applications",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "date" in data[0]
            assert "value" in data[0]
    
    def test_risk_distribution(self, auth_headers):
        """Test risk distribution analysis"""
        response = client.get(
            "/api/v1/analytics/risk-distribution",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "risk_level" in data[0]
            assert "count" in data[0]
            assert "percentage" in data[0]

class TestAdminAPI:
    """Test administration endpoints"""
    
    @pytest.fixture
    def admin_headers(self):
        """Get admin authentication headers"""
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": ADMIN_USER_DATA["username"],
                "password": ADMIN_USER_DATA["password"]
            }
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_system_health(self, admin_headers):
        """Test system health endpoint"""
        response = client.get(
            "/api/v1/admin/health",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "database_status" in data
    
    def test_list_users(self, admin_headers):
        """Test listing users"""
        response = client.get(
            "/api/v1/admin/users",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_system_configuration(self, admin_headers):
        """Test getting system configuration"""
        response = client.get(
            "/api/v1/admin/config",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "max_file_size_mb" in data
        assert "allowed_file_types" in data
    
    def test_admin_unauthorized(self):
        """Test admin endpoints without proper authorization"""
        # Try with regular user token (if implemented)
        response = client.get("/api/v1/admin/health")
        assert response.status_code == 401

class TestHealthAPI:
    """Test health and monitoring endpoints"""
    
    def test_basic_health_check(self):
        """Test basic health check (no auth required)"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_detailed_health_check(self):
        """Test detailed health check"""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "checks" in data
        assert "response_time_ms" in data
    
    def test_readiness_check(self):
        """Test readiness probe"""
        response = client.get("/api/v1/health/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    def test_liveness_check(self):
        """Test liveness probe"""
        response = client.get("/api/v1/health/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        # Should return Prometheus format
        assert "http_requests_total" in response.text
    
    def test_version_info(self):
        """Test version information"""
        response = client.get("/api/v1/health/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "build_date" in data

class TestSecurityFeatures:
    """Test security features and middleware"""
    
    def test_rate_limiting(self):
        """Test rate limiting (basic test)"""
        # Make multiple requests quickly
        responses = []
        for i in range(5):
            response = client.get("/health")
            responses.append(response)
        
        # All should succeed for basic health check
        for response in responses:
            assert response.status_code == 200
        
        # Check rate limit headers
        last_response = responses[-1]
        assert "X-RateLimit-Limit" in last_response.headers
        assert "X-RateLimit-Remaining" in last_response.headers
    
    def test_security_headers(self):
        """Test security headers are present"""
        response = client.get("/health")
        
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_cors_headers(self):
        """Test CORS configuration"""
        # Make an OPTIONS request
        response = client.options(
            "/api/v1/health/liveness",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should have CORS headers for allowed origin
        assert response.status_code in [200, 204]
    
    def test_input_validation(self):
        """Test input validation middleware"""
        # Try with potentially malicious input
        response = client.get("/api/v1/applications/?search=<script>alert('xss')</script>")
        
        # Should either block or sanitize
        assert response.status_code in [400, 401]  # 401 because no auth, 400 if blocked

class TestErrorHandling:
    """Test error handling and responses"""
    
    def test_404_error(self):
        """Test 404 error handling"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 error handling"""
        response = client.patch("/health")  # PATCH not allowed on health endpoint
        assert response.status_code == 405
    
    def test_validation_error(self):
        """Test validation error handling"""
        # Try to create application with invalid data
        response = client.post(
            "/api/v1/applications/",
            json={"invalid": "data"}
        )
        # Should get 401 (unauthorized) or 422 (validation error)
        assert response.status_code in [401, 422]

class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema generation"""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
    
    def test_swagger_ui(self):
        """Test Swagger UI availability"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_ui(self):
        """Test ReDoc UI availability"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

# Performance and load testing helpers
class TestPerformance:
    """Basic performance tests"""
    
    def test_health_check_performance(self):
        """Test health check response time"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])