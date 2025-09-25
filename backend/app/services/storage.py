"""
File storage abstraction layer supporting local filesystem and S3-compatible storage
"""
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional
import aiofiles
import aiofiles.os
from datetime import datetime
import uuid


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def store_file(self, file_content: BinaryIO, file_path: str) -> str:
        """Store file and return the storage path"""
        pass
    
    @abstractmethod
    async def retrieve_file(self, file_path: str) -> bytes:
        """Retrieve file content by path"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file by path"""
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        pass
    
    @abstractmethod
    async def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        pass


class LocalFileSystemStorage(StorageBackend):
    """Local filesystem storage implementation"""
    
    def __init__(self, base_path: str = "./uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, file_path: str) -> Path:
        """Get full filesystem path"""
        return self.base_path / file_path
    
    async def store_file(self, file_content: BinaryIO, file_path: str) -> str:
        """Store file to local filesystem"""
        full_path = self._get_full_path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'wb') as f:
            content = file_content.read()
            await f.write(content)
        
        return str(full_path.relative_to(self.base_path))
    
    async def retrieve_file(self, file_path: str) -> bytes:
        """Retrieve file from local filesystem"""
        full_path = self._get_full_path(file_path)
        if not await aiofiles.os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        async with aiofiles.open(full_path, 'rb') as f:
            return await f.read()
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local filesystem"""
        full_path = self._get_full_path(file_path)
        try:
            if await aiofiles.os.path.exists(full_path):
                await aiofiles.os.remove(full_path)
                return True
            return False
        except Exception:
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in local filesystem"""
        full_path = self._get_full_path(file_path)
        return await aiofiles.os.path.exists(full_path)
    
    async def get_file_size(self, file_path: str) -> int:
        """Get file size from local filesystem"""
        full_path = self._get_full_path(file_path)
        if not await aiofiles.os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = await aiofiles.os.stat(full_path)
        return stat.st_size


class S3CompatibleStorage(StorageBackend):
    """S3-compatible storage implementation (placeholder for future implementation)"""
    
    def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None, 
                 access_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        # TODO: Initialize S3 client when needed
    
    async def store_file(self, file_content: BinaryIO, file_path: str) -> str:
        """Store file to S3-compatible storage"""
        # TODO: Implement S3 storage
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def retrieve_file(self, file_path: str) -> bytes:
        """Retrieve file from S3-compatible storage"""
        # TODO: Implement S3 retrieval
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from S3-compatible storage"""
        # TODO: Implement S3 deletion
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in S3-compatible storage"""
        # TODO: Implement S3 file existence check
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def get_file_size(self, file_path: str) -> int:
        """Get file size from S3-compatible storage"""
        # TODO: Implement S3 file size check
        raise NotImplementedError("S3 storage not yet implemented")


class StorageManager:
    """Storage manager that handles file operations with different backends"""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend
    
    def generate_file_path(self, filename: str, document_type: str) -> str:
        """Generate a unique file path for storage"""
        # Create date-based directory structure
        now = datetime.utcnow()
        date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix
        unique_filename = f"{file_id}{file_extension}"
        
        return f"{document_type}/{date_path}/{unique_filename}"
    
    async def store_document(self, file_content: BinaryIO, filename: str, 
                           document_type: str) -> str:
        """Store document and return storage path"""
        file_path = self.generate_file_path(filename, document_type)
        stored_path = await self.storage_backend.store_file(file_content, file_path)
        return stored_path
    
    async def retrieve_document(self, file_path: str) -> bytes:
        """Retrieve document content"""
        return await self.storage_backend.retrieve_file(file_path)
    
    async def delete_document(self, file_path: str) -> bool:
        """Delete document"""
        return await self.storage_backend.delete_file(file_path)
    
    async def document_exists(self, file_path: str) -> bool:
        """Check if document exists"""
        return await self.storage_backend.file_exists(file_path)
    
    async def get_document_size(self, file_path: str) -> int:
        """Get document size"""
        return await self.storage_backend.get_file_size(file_path)


# Factory function to create storage manager based on configuration
def create_storage_manager(storage_type: str = "local", **kwargs) -> StorageManager:
    """Create storage manager with specified backend"""
    if storage_type == "local":
        base_path = kwargs.get("base_path", "./uploads")
        backend = LocalFileSystemStorage(base_path)
    elif storage_type == "s3":
        backend = S3CompatibleStorage(
            bucket_name=kwargs.get("bucket_name"),
            endpoint_url=kwargs.get("endpoint_url"),
            access_key=kwargs.get("access_key"),
            secret_key=kwargs.get("secret_key")
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    return StorageManager(backend)