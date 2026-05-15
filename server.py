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

from app.agents.engine import AgentEngine
from app.config import (
    CONTEXT_WINDOW,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)
from app.memory.supabase_store import (
    get_conversation_history,
    store_agent_trace,
    store_conversation,
)
from app.models.schemas import AgentRequest, ChatMessage, ChatRequest, WorkflowRequest
from app.tools.registry import ToolRegistry
from app.workflows.orchestrator import WorkflowOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("chat_server")

app = FastAPI(
    title="Gemma Agentic Business AI",
    version="2.0.0",
    description="Agentic reasoning API for business SaaS — powered by Gemma via Ollama",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    timeout=120.0,
    max_retries=2,
)

try:
    _enc = tiktoken.get_encoding("cl100k_base")
    count_tokens = lambda t: len(_enc.encode(t))

    def truncate(text: str, limit: int) -> tuple[str, int]:
        toks = _enc.encode(text)
        if len(toks) <= limit:
            return text, len(toks)
        return _enc.decode(toks[:limit]), limit

    logger.info("tiktoken ready")
except Exception:
    count_tokens = lambda t: max(1, len(t) // 4)

    def truncate(text: str, limit: int) -> tuple[str, int]:
        cap = limit * 4
        t = text[:cap] if len(text) > cap else text
        return t, count_tokens(t)

    logger.warning("tiktoken unavailable — char fallback active")

agent_engine = AgentEngine()
workflow_orchestrator = WorkflowOrchestrator()
tool_registry = ToolRegistry()


def make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def usage_block(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def non_stream_body(
    cmpl_id: str, content: str, finish_reason: str,
    prompt_tokens: int, completion_tokens: int,
    tool_calls: list | None = None,
) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage_block(prompt_tokens, completion_tokens),
    }


def stream_chunk(
    cmpl_id: str, delta: dict, finish_reason: str | None,
    *, usage: dict | None = None,
) -> str:
    body: dict = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


async def _stream(
    cmpl_id: str, messages: list[dict], max_tokens: int,
    temperature: float, prompt_tokens: int,
    tools: list[Any] | None = None,
    tool_choice: Any | None = None,
) -> AsyncGenerator[str, None]:
    yield stream_chunk(cmpl_id, {"role": "assistant", "content": ""}, None)

    text = ""
    comp_tokens = 0
    stop_reason = "stop"

    try:
        kwargs = {
            "model": MODEL, 
            "messages": messages,
            "max_tokens": max_tokens, 
            "temperature": temperature, 
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        async with await client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                choice = chunk.choices[0]
                delta = {}
                piece = choice.delta.content or ""
                if piece:
                    text += piece
                    comp_tokens += count_tokens(piece)
                    delta["content"] = piece
                
                if getattr(choice.delta, "tool_calls", None):
                    delta["tool_calls"] = [
                        {
                            "id": tc.id,
                            "index": tc.index,
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", None),
                                "arguments": getattr(tc.function, "arguments", None)
                            }
                        }
                        for tc in choice.delta.tool_calls
                    ]
                    # Remove None values
                    for tc in delta["tool_calls"]:
                        tc["function"] = {k: v for k, v in tc["function"].items() if v is not None}
                        if not tc["function"]:
                            del tc["function"]

                if delta:
                    yield stream_chunk(cmpl_id, delta, None)
                    
                if choice.finish_reason:
                    stop_reason = "length" if choice.finish_reason == "length" else "stop"
    except APITimeoutError:
        logger.error("[%s] timeout", cmpl_id)
        yield stream_chunk(cmpl_id, {}, "stop", usage=usage_block(prompt_tokens, comp_tokens))
        yield "data: [DONE]\n\n"
        return
    except APIError as exc:
        logger.error("[%s] API error: %s", cmpl_id, exc)
        yield stream_chunk(cmpl_id, {}, "stop", usage=usage_block(prompt_tokens, comp_tokens))
        yield "data: [DONE]\n\n"
        return

    yield stream_chunk(cmpl_id, {}, stop_reason, usage=usage_block(prompt_tokens, comp_tokens))
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(422, detail={
            "error": {
                "message": "Last message must have role 'user'.",
                "type": "invalid_request_error",
                "code": "invalid_messages",
            }
        })

    cmpl_id = make_completion_id()
    msgs = [m.model_dump() for m in req.messages]
    msgs[-1]["content"], prompt_tokens = truncate(msgs[-1]["content"], MAX_PROMPT_TOKENS)

    if req.stream:
        return StreamingResponse(
            _stream(
                cmpl_id, msgs, req.max_tokens, req.temperature, prompt_tokens,
                tools=getattr(req, "tools", None),
                tool_choice=getattr(req, "tool_choice", None)
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        kwargs = {
            "model": MODEL, 
            "messages": msgs,
            "max_tokens": req.max_tokens, 
            "temperature": req.temperature,
        }
        if getattr(req, "tools", None):
            kwargs["tools"] = req.tools
        if getattr(req, "tool_choice", None):
            kwargs["tool_choice"] = req.tool_choice

        resp = await client.chat.completions.create(**kwargs)
    except APITimeoutError:
        raise HTTPException(504, detail={"error": {"message": "Model timed out.", "type": "timeout_error"}})
    except APIError as exc:
        raise HTTPException(502, detail={"error": {"message": str(exc), "type": "api_error"}})

    msg = resp.choices[0].message
    content = msg.content or ""
    comp_tokens = count_tokens(content)
    finish = resp.choices[0].finish_reason or "stop"
    
    tool_calls = None
    if getattr(msg, "tool_calls", None):
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in msg.tool_calls
        ]

    return JSONResponse(content=non_stream_body(cmpl_id, content, finish, prompt_tokens, comp_tokens, tool_calls=tool_calls))


@app.post("/v1/agent/run")
async def agent_run(req: AgentRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(422, detail={"error": "Last message must have role 'user'."})

    result = await agent_engine.run(req)

    if req.session_id:
        user_msg = req.messages[-1].content
        await store_conversation(req.session_id, req.org_id, "user", user_msg)
        await store_conversation(req.session_id, req.org_id, "assistant", result.answer, {
            "agent_id": result.id,
            "tool_calls": result.tool_calls_made,
            "iterations": result.iterations_used,
        })
        await store_agent_trace(
            result.id, req.org_id,
            [s.model_dump() for s in result.steps],
            result.answer,
        )

    return JSONResponse(content=result.model_dump())


@app.post("/v1/agent/stream")
async def agent_stream(req: AgentRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(422, detail={"error": "Last message must have role 'user'."})

    async def _agent_stream_generator():
        yield stream_chunk("agent-stream", {"role": "assistant", "content": ""}, None)
        yield stream_chunk("agent-stream", {"content": "[Agent reasoning started...]\n\n"}, None)

        result = await agent_engine.run(req)

        for step in result.steps:
            step_text = f"**Step {step.iteration + 1}** ({step.action.value}):\n"
            if step.thought:
                step_text += f"  Thought: {step.thought[:200]}\n"
            if step.tool_call:
                step_text += f"  Tool: {step.tool_call.tool_name}\n"
            if step.observation:
                step_text += f"  Observation: {step.observation[:200]}\n"
            step_text += "\n"
            yield stream_chunk("agent-stream", {"content": step_text}, None)

        yield stream_chunk("agent-stream", {"content": f"\n**Final Answer:**\n{result.answer}"}, None)
        yield stream_chunk("agent-stream", {}, "stop", usage=usage_block(0, 0))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _agent_stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/workflow/run")
async def workflow_run(req: WorkflowRequest):
    result = await workflow_orchestrator.execute(req)
    return JSONResponse(content=result)


@app.get("/v1/workflow/list")
async def workflow_list():
    return JSONResponse(content=WorkflowOrchestrator.list_workflows())


@app.get("/v1/tools")
async def list_tools():
    tools = [t.schema() for t in tool_registry.all()]
    return JSONResponse(content={"tools": tools, "count": len(tools)})


@app.get("/v1/conversations/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    history = await get_conversation_history(session_id, limit)
    return JSONResponse(content={"session_id": session_id, "messages": history})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "context_window": CONTEXT_WINDOW,
        "tools_registered": len(tool_registry.all()),
        "workflows": list(WorkflowOrchestrator.list_workflows().keys()),
    }
