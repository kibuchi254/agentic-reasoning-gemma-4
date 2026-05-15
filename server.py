from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator

import tiktoken
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI, APIError, APITimeoutError
from pydantic import BaseModel, Field

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("chat_server")

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── async client ──────────────────────────────────────────────────────────────
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=120.0,
    max_retries=2,
)

# ── tokenizer ─────────────────────────────────────────────────────────────────
try:
    _enc         = tiktoken.get_encoding("cl100k_base")
    count_tokens = lambda t: len(_enc.encode(t))

    def truncate(text: str, limit: int) -> tuple[str, int]:
        toks = _enc.encode(text)
        if len(toks) <= limit:
            return text, len(toks)
        return _enc.decode(toks[:limit]), limit

    logger.info("tiktoken ready")
except Exception:
    _enc         = None
    count_tokens = lambda t: max(1, len(t) // 4)

    def truncate(text: str, limit: int) -> tuple[str, int]:
        cap = limit * 4
        t   = text[:cap] if len(text) > cap else text
        return t, count_tokens(t)

    logger.warning("tiktoken unavailable – char fallback active")

# ── config ────────────────────────────────────────────────────────────────────
MODEL             = "gemma4:latest"
CONTEXT_WINDOW    = 8_192
MAX_PROMPT_TOKENS = 6_144
MAX_NEW_TOKENS    = 2_048


# ── schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    model: str             = MODEL
    messages: list[Message]
    stream: bool           = False
    max_tokens: int        = Field(MAX_NEW_TOKENS, ge=1, le=MAX_NEW_TOKENS)
    temperature: float     = Field(0.7, ge=0.0, le=2.0)


# ── OpenAI-spec response builders ─────────────────────────────────────────────

def make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"

def usage_block(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      prompt_tokens + completion_tokens,
    }

def non_stream_body(
    cmpl_id: str,
    content: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Mirrors the exact shape of an OpenAI /v1/chat/completions response."""
    return {
        "id":      cmpl_id,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL,
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
                "logprobs":      None,
            }
        ],
        "usage": usage_block(prompt_tokens, completion_tokens),
    }

def stream_chunk(
    cmpl_id: str,
    delta: dict,
    finish_reason: str | None,
    *,
    usage: dict | None = None,
) -> str:
    """
    Mirrors OpenAI streaming chunks exactly:
      data: {"id":…,"object":"chat.completion.chunk","created":…,"model":…,
             "choices":[{"index":0,"delta":{…},"finish_reason":…}]}
    Final chunk also carries usage (matches OpenAI stream_options behaviour).
    """
    body: dict = {
        "id":      cmpl_id,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   MODEL,
        "choices": [
            {
                "index":         0,
                "delta":         delta,
                "finish_reason": finish_reason,
                "logprobs":      None,
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


# ── streaming generator ───────────────────────────────────────────────────────
async def _stream(
    cmpl_id: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    prompt_tokens: int,
) -> AsyncGenerator[str, None]:

    # ── role chunk (OpenAI sends this first) ──────────────────────────────────
    yield stream_chunk(cmpl_id, {"role": "assistant", "content": ""}, None)

    text        = ""
    comp_tokens = 0
    stop_reason = "stop"

    try:
        async with await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ) as stream:
            async for chunk in stream:
                choice = chunk.choices[0]
                piece  = choice.delta.content or ""

                if piece:
                    text        += piece
                    comp_tokens += count_tokens(piece)
                    yield stream_chunk(cmpl_id, {"content": piece}, None)

                if choice.finish_reason:
                    stop_reason = (
                        "length" if choice.finish_reason == "length" else "stop"
                    )

    except APITimeoutError:
        logger.error("[%s] timeout", cmpl_id)
        yield stream_chunk(cmpl_id, {}, "stop",
                           usage=usage_block(prompt_tokens, comp_tokens))
        yield "data: [DONE]\n\n"
        return

    except APIError as exc:
        logger.error("[%s] API error: %s", cmpl_id, exc)
        yield stream_chunk(cmpl_id, {}, "stop",
                           usage=usage_block(prompt_tokens, comp_tokens))
        yield "data: [DONE]\n\n"
        return

    # ── final chunk: finish_reason + usage ────────────────────────────────────
    yield stream_chunk(
        cmpl_id, {}, stop_reason,
        usage=usage_block(prompt_tokens, comp_tokens),
    )
    # ── OpenAI terminator ─────────────────────────────────────────────────────
    yield "data: [DONE]\n\n"


# ── /v1/chat/completions ──────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(422, detail={
            "error": {
                "message": "Last message must have role 'user'.",
                "type":    "invalid_request_error",
                "code":    "invalid_messages",
            }
        })

    cmpl_id = make_completion_id()
    msgs    = [m.model_dump() for m in req.messages]
    msgs[-1]["content"], prompt_tokens = truncate(msgs[-1]["content"], MAX_PROMPT_TOKENS)

    # ── streaming ─────────────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream(cmpl_id, msgs, req.max_tokens, req.temperature, prompt_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── non-streaming ─────────────────────────────────────────────────────────
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except APITimeoutError:
        raise HTTPException(504, detail={
            "error": {"message": "Model timed out.", "type": "timeout_error"}
        })
    except APIError as exc:
        raise HTTPException(502, detail={
            "error": {"message": str(exc), "type": "api_error"}
        })

    content     = resp.choices[0].message.content
    comp_tokens = count_tokens(content)
    finish      = resp.choices[0].finish_reason or "stop"

    return JSONResponse(
        content=non_stream_body(cmpl_id, content, finish, prompt_tokens, comp_tokens)
    )


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "context_window": CONTEXT_WINDOW}