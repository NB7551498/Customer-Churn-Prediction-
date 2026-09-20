"""
Unit and integration tests for ChurnGuard AI authentication and RBAC.
"""

from datetime import timedelta
import pytest
from fastapi import HTTPException
from app.auth import (
    RoleChecker,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification():
    """Verify bcrypt hash generation and validation."""
    pwd = "SecurePassword123!"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_create_and_decode():
    """Verify creation and decoding of valid JWT access tokens."""
    payload = {"sub": "analyst@churnguard.ai", "role": "analyst"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=30))
    decoded = decode_access_token(token)
    assert decoded["sub"] == "analyst@churnguard.ai"
    assert decoded["role"] == "analyst"
    assert "exp" in decoded


def test_jwt_expired_token_raises():
    """Verify expired JWT access tokens trigger 401 HTTPException."""
    payload = {"sub": "viewer@churnguard.ai", "role": "viewer"}
    token = create_access_token(payload, expires_delta=timedelta(seconds=-10))
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_jwt_tampered_token_raises():
    """Verify tampered JWT tokens trigger 401 HTTPException."""
    payload = {"sub": "admin@churnguard.ai", "role": "admin"}
    token = create_access_token(payload)
    tampered = token[:-4] + "abcd"
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(tampered)
    assert exc_info.value.status_code == 401


def test_role_checker_allows_authorized_role():
    """Verify RoleChecker permits users with required role."""
    checker = RoleChecker(["admin", "analyst"])
    user = {"username": "admin@churnguard.ai", "role": "admin"}
    result = checker(user=user)
    assert result == user


def test_role_checker_rejects_unauthorized_role():
    """Verify RoleChecker blocks users lacking authorized role with 403."""
    checker = RoleChecker(["admin"])
    user = {"username": "viewer@churnguard.ai", "role": "viewer"}
    with pytest.raises(HTTPException) as exc_info:
        checker(user=user)
    assert exc_info.value.status_code == 403
    assert "access forbidden" in exc_info.value.detail.lower()
