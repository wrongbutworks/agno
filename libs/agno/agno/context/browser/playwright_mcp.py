"""PlaywrightMCPBackend — browser automation via Playwright's MCP server.

Runs `npx @playwright/mcp@latest` as a subprocess and exposes browser
tools (navigate, snapshot, screenshot, click, type) to the calling agent.

Default is readonly=True, which excludes interaction tools (click, type,
etc.) for safe browsing. Set readonly=False for full browser control.

Requires Node.js 18+ (npx downloads the package on first run).
"""

from __future__ import annotations

from typing import Any, Literal

from agno.context.backend import ContextBackend
from agno.context.provider import Status
from agno.utils.log import log_warning

_INTERACTION_TOOLS: list[str] = [
    "browser_click",
    "browser_close",
    "browser_drag",
    "browser_drop",
    "browser_evaluate",
    "browser_file_upload",
    "browser_fill_form",
    "browser_handle_dialog",
    "browser_hover",
    "browser_press_key",
    "browser_run_code_unsafe",
    "browser_select_option",
    "browser_type",
]


class PlaywrightMCPBackend(ContextBackend):
    """Backend for `BrowserContextProvider` that runs Playwright's MCP server."""

    def __init__(
        self,
        *,
        headless: bool = True,
        browser: Literal["chromium", "firefox", "webkit"] = "chromium",
        readonly: bool = True,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        tool_name_prefix: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.headless = headless
        self.browser = browser
        self.readonly = readonly
        self.include_tools = include_tools
        self.exclude_tools = exclude_tools
        self.tool_name_prefix = tool_name_prefix
        self.timeout_seconds = timeout_seconds
        self._mcp_tools: Any = None

    def status(self) -> Status:
        mode = "readonly" if self.readonly else "read-write"
        if self._mcp_tools is None:
            return Status(ok=True, detail=f"playwright-mcp ({self.browser}, {mode}, not connected)")
        if getattr(self._mcp_tools, "initialized", False):
            return Status(ok=True, detail=f"playwright-mcp ({self.browser}, {mode}, connected)")
        return Status(ok=True, detail=f"playwright-mcp ({self.browser}, {mode}, pending)")

    async def astatus(self) -> Status:
        return self.status()

    def get_tools(self) -> list:
        if self._mcp_tools is None:
            self._mcp_tools = self._build_tools()
        return [self._mcp_tools]

    def _build_tools(self) -> Any:
        from mcp import StdioServerParameters

        from agno.tools.mcp import MCPTools

        cmd_args = ["@playwright/mcp@latest"]
        if self.headless:
            cmd_args.append("--headless")
        if self.browser != "chromium":
            cmd_args.append(f"--browser={self.browser}")

        exclude = (self.exclude_tools or []) + (_INTERACTION_TOOLS if self.readonly else [])

        return MCPTools(
            server_params=StdioServerParameters(command="npx", args=cmd_args),
            transport="stdio",
            include_tools=self.include_tools,
            exclude_tools=exclude if exclude else None,
            tool_name_prefix=self.tool_name_prefix,
            timeout_seconds=self.timeout_seconds,
        )

    async def asetup(self) -> None:
        """Start the Playwright MCP server and connect.

        On failure, logs a warning; the browser backend will be
        unavailable until the next restart.
        """
        if self._mcp_tools is None:
            self._mcp_tools = self._build_tools()
        if getattr(self._mcp_tools, "initialized", False):
            return
        try:
            await self._mcp_tools._connect()
        except Exception as exc:
            log_warning(f"PlaywrightMCPBackend setup failed — {type(exc).__name__}: {exc}.")
            self._mcp_tools = None

    async def aclose(self) -> None:
        """Stop the MCP server and clear cached state."""
        tools = self._mcp_tools
        self._mcp_tools = None
        if tools is None:
            return
        try:
            await tools.close()
        except Exception as exc:
            log_warning(f"PlaywrightMCPBackend close raised {type(exc).__name__}: {exc}")
