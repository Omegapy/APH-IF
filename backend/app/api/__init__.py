"""API package providing router organization for the backend service."""

from .lifecycle import register_shutdown, register_startup

__all__ = [
    "register_shutdown",
    "register_startup",
]
