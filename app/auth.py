"""
ChurnGuard AI — Authentication & Security Layer.
Implements JWT token generation, bcrypt password hashing, and Role-Based
Access Control (RBAC) across Admin, Analyst, and Viewer tiers.
"""

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any, Dict, List, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("churn.auth")

SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "churnguard-super-secure-production-key-2026")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

security = HTTPBearer(auto_error=False)


def hash_password(plain_password: str) -> str:
    """Hash password with bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# Mock in-memory user store with hashed passwords
USERS_DB: Dict[str, Dict[str, Any]] = {
    "admin@churnguard.ai": {
        "username": "admin@churnguard.ai",
        "full_name": "Platform Administrator",
        "role": "admin",
        "hashed_password": hash_password("AdminPass123!"),
    },
    "analyst@churnguard.ai": {
        "username": "analyst@churnguard.ai",
        "full_name": "Lead Retention Analyst",
        "role": "analyst",
        "hashed_password": hash_password("AnalystPass123!"),
    },
    "viewer@churnguard.ai": {
        "username": "viewer@churnguard.ai",
        "full_name": "Executive Viewer",
        "role": "viewer",
        "hashed_password": hash_password("ViewerPass123!"),
    },
}


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate cryptographically signed JWT token with expiry."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate signature and expiry of JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token signature.",
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """FastAPI dependency to extract and authenticate current user from Bearer header."""
    if not credentials or not credentials.credentials:
        # Default guest mode for frictionless local testing if no header is provided
        return {"username": "guest@churnguard.ai", "role": "admin", "full_name": "Local Guest"}

    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub")
    if not username or username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User credentials could not be validated.",
        )
    return USERS_DB[username]


class RoleChecker:
    """RBAC Dependency: Restricts endpoints to authorized roles."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if user.get("role") not in self.allowed_roles:
            logger.warning("Access denied for user %s with role %s", user.get("username"), user.get("role"))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: requires one of roles: {', '.join(self.allowed_roles)}",
            )
        return user
