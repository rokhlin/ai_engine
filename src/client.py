"""
src/client.py
Export unified client for Media Cataloger AI Engine and Web API.
"""
from media_cataloger_client import (
    MediaCatalogerClient,
    CatalogerApiError,
    AuthenticationError,
    PermissionDeniedError,
    VaultLockedError,
    InvalidRequestError
)

__all__ = [
    "MediaCatalogerClient",
    "CatalogerApiError",
    "AuthenticationError",
    "PermissionDeniedError",
    "VaultLockedError",
    "InvalidRequestError"
]
