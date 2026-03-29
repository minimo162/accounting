"""Structured logging helpers for request-level observability."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
import os
import uuid
from typing import Any, Iterator


_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_MONITOR_CASE_ID: ContextVar[str] = ContextVar("monitor_case_id", default="")


def generate_request_id() -> str:
    return uuid.uuid4().hex


def get_request_id() -> str:
    return _REQUEST_ID.get()


def get_monitor_case_id() -> str:
    return _MONITOR_CASE_ID.get()


@contextmanager
def request_context(request_id: str, monitor_case_id: str = "") -> Iterator[None]:
    request_token = _REQUEST_ID.set(request_id)
    monitor_token = _MONITOR_CASE_ID.set(monitor_case_id)
    try:
        yield
    finally:
        _MONITOR_CASE_ID.reset(monitor_token)
        _REQUEST_ID.reset(request_token)


def question_sha1(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def structured_log(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "request_id": fields.pop("request_id", None) or get_request_id() or None,
        "monitor_case_id": fields.pop("monitor_case_id", None) or get_monitor_case_id() or None,
        "service": os.getenv("OBS_SERVICE") or os.getenv("K_SERVICE") or None,
        "revision": os.getenv("K_REVISION") or None,
        "environment": os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or None,
        **fields,
    }
    compact = {key: value for key, value in payload.items() if value is not None}
    logger.log(level, json.dumps(compact, ensure_ascii=False, sort_keys=True))
