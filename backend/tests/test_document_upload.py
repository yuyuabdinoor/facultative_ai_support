"""
Integration tests for document upload functionality
"""
import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import io

from app.main import app
from app.core.database import get_db, Base
from app.models.database import Document as DocumentModel


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override upload directory in settings
        from app.core.config import settings
        original_upload_dir = settings.upload_dir
        settings.upload_dir = temp_dir
        yield temp_dir
        settings.upload_dir = original_upload_dir


@pytest.fixture
def sample_pdf_content():
    """Create a simple PDF content for testing"""
    # This is a minimal PDF content
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
300
%%EOF"""


@pytest.fixture
def sample_excel_content():
    """Create sample Excel content"""
    import pandas as pd
    
    # Create a simple Excel file in memory
    df = pd.DataFrame({
        'Asset Type': ['Building', 'Equipment'],
        'Value': [1000000, 500000],
        'Location': ['New York', 'California']
    })
    
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


class TestDocumentUpload:
    """Test document upload functionality"""
    
    def test_upload_pdf_document(self, client, temp_upload_dir, sample_pdf_content):
        """Test uploading a PDF document"""
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert data["filename"] == "test.pdf"
        assert data["document_type"] == "pdf"
        assert data["processed"] is False
        assert "upload_timestamp" in data
        assert "metadata" in data
    
    def test_upload_excel_document(self, client, temp_upload_dir, sample_excel_content):
        """Test uploading an Excel document"""
        files = {"file": ("test.xlsx", sample_excel_content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["filename"] == "test.xlsx"
        assert data["document_type"] == "excel"
        assert "sheet_count" in data["metadata"]
    
    def test_upload_with_application_id(self, client, temp_upload_dir, sample_pdf_content):
        """Test uploading document with application ID"""
        application_id = "test-app-123"
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        
        response = client.post(
            f"/api/v1/documents/upload?application_id={application_id}",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["application_id"] == application_id
    
    def test_upload_invalid_file_type(self, client, temp_upload_dir):
        """Test uploading unsupported file type"""
        files = {"file": ("test.txt", b"Hello World", "text/plain")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "validation failed" in data["detail"]["message"].lower()
    
    def test_upload_empty_file(self, client, temp_upload_dir):
        """Test uploading empty file"""
        files = {"file": ("test.pdf", b"", "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 400
    
    def test_upload_oversized_file(self, client, temp_upload_dir):
        """Test uploading file that exceeds size limit"""
        # Create a large file (larger than PDF limit)
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        files = {"file": ("large.pdf", large_content, "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "size" in data["detail"]["message"].lower()


class TestDocumentValidation:
    """Test document validation functionality"""
    
    def test_validate_valid_pdf(self, client, sample_pdf_content):
        """Test validating a valid PDF"""
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        
        response = client.post("/api/v1/documents/validate", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)
    
    def test_validate_invalid_file(self, client):
        """Test validating invalid file"""
        files = {"file": ("test.txt", b"Hello World", "text/plain")}
        
        response = client.post("/api/v1/documents/validate", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0


class TestDocumentRetrieval:
    """Test document retrieval functionality"""
    
    def test_get_document_by_id(self, client, temp_upload_dir, sample_pdf_content):
        """Test retrieving document by ID"""
        # First upload a document
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        upload_response = client.post("/api/v1/documents/upload", files=files)
        document_id = upload_response.json()["id"]
        
        # Then retrieve it
        response = client.get(f"/api/v1/documents/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == document_id
        assert data["filename"] == "test.pdf"
    
    def test_get_nonexistent_document(self, client):
        """Test retrieving non-existent document"""
        response = client.get("/api/v1/documents/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_download_document(self, client, temp_upload_dir, sample_pdf_content):
        """Test downloading document content"""
        # Upload document
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        upload_response = client.post("/api/v1/documents/upload", files=files)
        document_id = upload_response.json()["id"]
        
        # Download document
        response = client.get(f"/api/v1/documents/{document_id}/download")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "attachment" in response.headers["content-disposition"]
    
    def test_get_document_status(self, client, temp_upload_dir, sample_pdf_content):
        """Test getting document status"""
        # Upload document
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        upload_response = client.post("/api/v1/documents/upload", files=files)
        document_id = upload_response.json()["id"]
        
        # Get status
        response = client.get(f"/api/v1/documents/{document_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == document_id
        assert "file_exists" in data
        assert "file_size" in data
        assert data["processed"] is False


class TestDocumentManagement:
    """Test document management operations"""
    
    def test_update_document(self, client, temp_upload_dir, sample_pdf_content):
        """Test updating document metadata"""
        # Upload document
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        upload_response = client.post("/api/v1/documents/upload", files=files)
        document_id = upload_response.json()["id"]
        
        # Update document
        update_data = {
            "processed": True,
            "metadata": {"custom_field": "custom_value"}
        }
        response = client.put(f"/api/v1/documents/{document_id}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["processed"] is True
        assert "custom_field" in data["metadata"]
    
    def test_delete_document(self, client, temp_upload_dir, sample_pdf_content):
        """Test deleting document"""
        # Upload document
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        upload_response = client.post("/api/v1/documents/upload", files=files)
        document_id = upload_response.json()["id"]
        
        # Delete document
        response = client.delete(f"/api/v1/documents/{document_id}")
        
        assert response.status_code == 200
        
        # Verify document is deleted
        get_response = client.get(f"/api/v1/documents/{document_id}")
        assert get_response.status_code == 404
    
    def test_list_documents(self, client, temp_upload_dir, sample_pdf_content):
        """Test listing documents"""
        # Upload multiple documents
        for i in range(3):
            files = {"file": (f"test{i}.pdf", sample_pdf_content, "application/pdf")}
            client.post("/api/v1/documents/upload", files=files)
        
        # List documents
        response = client.get("/api/v1/documents/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        assert all("id" in doc for doc in data)
    
    def test_list_documents_with_pagination(self, client, temp_upload_dir, sample_pdf_content):
        """Test listing documents with pagination"""
        # Upload documents
        for i in range(5):
            files = {"file": (f"test{i}.pdf", sample_pdf_content, "application/pdf")}
            client.post("/api/v1/documents/upload", files=files)
        
        # List with pagination
        response = client.get("/api/v1/documents/?skip=2&limit=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2


class TestBatchUpload:
    """Test batch upload functionality"""
    
    def test_batch_upload_success(self, client, temp_upload_dir, sample_pdf_content):
        """Test successful batch upload"""
        files = [
            ("files", ("test1.pdf", sample_pdf_content, "application/pdf")),
            ("files", ("test2.pdf", sample_pdf_content, "application/pdf"))
        ]
        
        response = client.post("/api/v1/documents/batch-upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all("id" in doc for doc in data)
    
    def test_batch_upload_with_invalid_file(self, client, temp_upload_dir, sample_pdf_content):
        """Test batch upload with one invalid file"""
        files = [
            ("files", ("test1.pdf", sample_pdf_content, "application/pdf")),
            ("files", ("test2.txt", b"Hello", "text/plain"))  # Invalid
        ]
        
        response = client.post("/api/v1/documents/batch-upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "validation failed" in data["detail"]["message"].lower()
    
    def test_batch_upload_too_many_files(self, client, temp_upload_dir, sample_pdf_content):
        """Test batch upload with too many files"""
        files = [
            ("files", (f"test{i}.pdf", sample_pdf_content, "application/pdf"))
            for i in range(15)  # More than limit of 10
        ]
        
        response = client.post("/api/v1/documents/batch-upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "maximum" in data["detail"].lower()


class TestDocumentReprocessing:
    """Test document reprocessing functionality"""
    
    def test_reprocess_document(self, client, temp_upload_dir, sample_pdf_content):
        """Test marking document for reprocessing"""
        # Upload and mark as processed
        files = {"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        upload_response = client.post("/api/v1/documents/upload", files=files)
        document_id = upload_response.json()["id"]
        
        # Mark as processed
        client.put(f"/api/v1/documents/{document_id}", json={"processed": True})
        
        # Reprocess
        response = client.post(f"/api/v1/documents/{document_id}/reprocess")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify it's marked as unprocessed
        doc_response = client.get(f"/api/v1/documents/{document_id}")
        assert doc_response.json()["processed"] is False