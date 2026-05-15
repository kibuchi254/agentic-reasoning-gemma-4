from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from app.config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger("memory")

_client = None


def _get_client():
    global _client
    if _client is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client initialized")
        except Exception as exc:
            logger.warning("Supabase init failed: %s — memory will be in-memory only", exc)
    return _client


async def store_conversation(
    session_id: str,
    org_id: str | None,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    client = _get_client()
    if client is None:
        logger.debug("No Supabase — skipping store_conversation")
        return False

    try:
        client.table("ai_conversations").insert({
            "session_id": session_id,
            "org_id": org_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        return True
    except Exception as exc:
        logger.error("store_conversation failed: %s", exc)
        return False


async def get_conversation_history(
    session_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    client = _get_client()
    if client is None:
        return []

    try:
        resp = (
            client.table("ai_conversations")
            .select("role, content, metadata, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(reversed(resp.data or []))
    except Exception as exc:
        logger.error("get_conversation_history failed: %s", exc)
        return []


async def store_agent_trace(
    agent_id: str,
    org_id: str | None,
    steps: list[dict],
    final_answer: str,
) -> bool:
    client = _get_client()
    if client is None:
        return False

    try:
        client.table("ai_agent_traces").insert({
            "agent_id": agent_id,
            "org_id": org_id,
            "steps": json.dumps(steps, default=str),
            "final_answer": final_answer,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        return True
    except Exception as exc:
        logger.error("store_agent_trace failed: %s", exc)
        return False


async def store_workflow_run(
    workflow_id: str,
    org_id: str | None,
    workflow_type: str,
    status: str,
    result: dict[str, Any] | None = None,
) -> bool:
    client = _get_client()
    if client is None:
        return False

    try:
        client.table("ai_workflow_runs").upsert({
            "workflow_id": workflow_id,
            "org_id": org_id,
            "workflow_type": workflow_type,
            "status": status,
            "result": json.dumps(result, default=str) if result else None,
            "updated_at": datetime.utcnow().isoformat(),
        }).execute()
        return True
    except Exception as exc:
        logger.error("store_workflow_run failed: %s", exc)
        return False
