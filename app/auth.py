# app/auth.py
import os
from datetime import datetime, timedelta
from typing import Optional, Dict

from jose import jwt, JWTError
from passlib.context import CryptContext

# Use Argon2 as primary (handles long passwords) with pbkdf2_sha256 as fallback.
PWD_CTX = CryptContext(schemes=["argon2", "pbkdf2_sha256"], deprecated="auto")

# JWT settings
SECRET = os.getenv("IDA_JWT_SECRET", "change-me-in-prod")
ALGO = "HS256"

# Access token short (15 minutes) — good UX but rotate by refresh token
ACCESS_EXPIRE_MINUTES = int(os.getenv("IDA_ACCESS_MIN", 15))
# Refresh token longer (7 days)
REFRESH_EXPIRE_DAYS = int(os.getenv("IDA_REFRESH_DAYS", 7))


def _ensure_str(pw: str) -> str:
    """Normalize password input to a string."""
    if pw is None:
        return ""
    if not isinstance(pw, str):
        try:
            return str(pw)
        except Exception:
            return ""
    return pw


def hash_password(pw: str) -> str:
    """
    Hash a password. Argon2 supports long inputs, so no manual truncation is required.
    """
    pw = _ensure_str(pw)
    return PWD_CTX.hash(pw)


def verify_password(pw: str, pw_hash: str) -> bool:
    pw = _ensure_str(pw)
    try:
        return PWD_CTX.verify(pw, pw_hash)
    except Exception:
        # If verification raises for any reason, return False (do not crash).
        return False


def _now_utc() -> datetime:
    return datetime.utcnow()


def create_access_token(data: dict, expires_minutes: int = ACCESS_EXPIRE_MINUTES) -> str:
    """
    Create a signed JWT access token with 'exp' claim and type 'access'.
    """
    to_encode = data.copy()
    expire = _now_utc() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire, "type": "access"})
    token = jwt.encode(to_encode, SECRET, algorithm=ALGO)
    return token


def create_refresh_token(data: dict, expires_days: int = REFRESH_EXPIRE_DAYS) -> str:
    """
    Create a signed JWT refresh token with 'exp' claim and type 'refresh'.
    """
    to_encode = data.copy()
    expire = _now_utc() + timedelta(days=expires_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    token = jwt.encode(to_encode, SECRET, algorithm=ALGO)
    return token


def decode_token(token: str) -> Optional[Dict]:
    """
    Decode and verify a JWT. Returns payload dict on success, otherwise None.
    """
    if not token or not isinstance(token, str):
        return None
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        return payload
    except JWTError:
        return None
