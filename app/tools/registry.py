from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("tool_registry")


class ToolContext:
    _current: ToolContext | None = None

    def __init__(self, servio_base_url: str, auth_token: str):
        self.servio_base_url = servio_base_url.rstrip("/")
        self.auth_token = auth_token

    def auth_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    @classmethod
    def get(cls) -> ToolContext:
        if cls._current is None:
            return cls("", "")
        return cls._current

    @classmethod
    def set(cls, ctx: ToolContext | None):
        cls._current = ctx


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        ...

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    _instance: ToolRegistry | None = None
    _tools: dict[str, Tool]

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._auto_register()
        return cls._instance

    def _auto_register(self):
        from app.tools.business_tools import (
            AnalyticsTool,
            CreateInvoiceTool,
            CRMSearchTool,
            CustomerHistoryTool,
            GenerateReportTool,
            GetInvoiceTool,
            ListInvoicesTool,
            SearchKnowledgeBaseTool,
            SendNotificationTool,
            UpdateCRAMTool,
        )

        for cls in [
            GetInvoiceTool,
            ListInvoicesTool,
            CreateInvoiceTool,
            GenerateReportTool,
            CRMSearchTool,
            CustomerHistoryTool,
            UpdateCRAMTool,
            SearchKnowledgeBaseTool,
            SendNotificationTool,
            AnalyticsTool,
        ]:
            tool = cls()
            self._tools[tool.name] = tool
            logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def describe(self, filter_names: list[str] | None = None) -> str:
        tools = self.all()
        if filter_names:
            tools = [t for t in tools if t.name in filter_names]
        lines = []
        for t in tools:
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)
