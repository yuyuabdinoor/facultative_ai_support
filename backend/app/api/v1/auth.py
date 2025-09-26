"""
Authentication API endpoints
"""
from datetime import timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from ...core.auth import (
    authenticate_user, create_access_token, create_refresh_token,
    verify_token, get_user, get_current_active_user, User, Token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    scopes: list[str]

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint to get access and refresh tokens
    
    Parameters:
    - username: User's username
    - password: User's password
    
    Returns:
    - access_token: JWT token for API access
    - refresh_token: Token for refreshing access token
    - token_type: Always "bearer"
    - expires_in: Token expiration time in seconds
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "scopes": user.scopes
        },
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "scopes": user.scopes
        }
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    
    Parameters:
    - refresh_token: Valid refresh token
    
    Returns:
    - New access and refresh tokens
    """
    token_data = verify_token(request.refresh_token, token_type="refresh")
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = get_user(token_data.username)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "scopes": user.scopes
        },
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "scopes": user.scopes
        }
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    
    Returns detailed information about the currently authenticated user
    """
    return UserResponse(**current_user.dict())

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout endpoint
    
    Note: Since we're using stateless JWT tokens, logout is handled client-side
    by discarding the tokens. In a production system, you might want to implement
    token blacklisting.
    """
    return {"message": "Successfully logged out"}

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user)
):
    """
    Change user password
    
    Parameters:
    - current_password: User's current password
    - new_password: New password to set
    """
    # In a real implementation, you would:
    # 1. Verify the current password
    # 2. Hash the new password
    # 3. Update the database
    # 4. Optionally invalidate existing tokens
    
    # For now, return success (mock implementation)
    return {"message": "Password changed successfully"}

@router.get("/permissions")
async def get_user_permissions(current_user: User = Depends(get_current_active_user)):
    """
    Get current user's permissions and scopes
    
    Returns:
    - User scopes and permissions
    - Available actions based on permissions
    """
    permissions = {
        "scopes": current_user.scopes,
        "is_superuser": current_user.is_superuser,
        "actions": {
            "can_read": "read" in current_user.scopes or current_user.is_superuser,
            "can_write": "write" in current_user.scopes or current_user.is_superuser,
            "can_admin": "admin" in current_user.scopes or current_user.is_superuser,
            "can_upload_documents": "write" in current_user.scopes or current_user.is_superuser,
            "can_delete_documents": "write" in current_user.scopes or current_user.is_superuser,
            "can_manage_users": "admin" in current_user.scopes or current_user.is_superuser,
            "can_view_analytics": "read" in current_user.scopes or current_user.is_superuser,
            "can_export_data": "read" in current_user.scopes or current_user.is_superuser
        }
    }
    
    return permissions

@router.post("/validate-token")
async def validate_token(current_user: User = Depends(get_current_active_user)):
    """
    Validate current token
    
    This endpoint can be used by frontend applications to check
    if the current token is still valid.
    """
    return {
        "valid": True,
        "user": UserResponse(**current_user.dict()),
        "message": "Token is valid"
    }