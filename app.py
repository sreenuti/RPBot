"""Vercel / default FastAPI entrypoint (re-exports ui.app)."""

from ui.app import app

__all__ = ["app"]
