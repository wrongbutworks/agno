from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from agno.context.backend import ContextBackend
from agno.context.provider import Status
from agno.utils.log import log_info, log_warning

BrowserType = Literal["chromium", "firefox", "webkit"]

# Tools that modify page state — excluded when provider has write=False
INTERACTION_TOOLS: frozenset[str] = frozenset(
    [
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_select_option",
        "browser_drag",
        "browser_file_upload",
        "browser_handle_dialog",
        "browser_press_key",
        "browser_evaluate",
    ]
)


class PlaywrightMCPBackend(ContextBackend):
    """Browser automation via Playwright's official MCP server.

    Runs `npx @playwright/mcp` as a subprocess and exposes browser tools
    (navigate, snapshot, screenshot, click, type, etc.) to the calling
    agent. Uses the accessibility tree by default, which is ~4x more
    token-efficient than vision-based approaches.

    Requires Node.js 18+ (npx downloads the package on first run).
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        browser: BrowserType = "chromium",
        version: str = "latest",
        env: Mapping[str, str] | None = None,
        include_tools: Sequence[str] | None = None,
        exclude_tools: Sequence[str] | None = None,
        tool_name_prefix: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.headless = headless
        self.browser: BrowserType = browser
        self.version = version
        self.env = dict(env) if env is not None else None
        self.include_tools = list(include_tools) if include_tools is not None else None
        self.exclude_tools = list(exclude_tools) if exclude_tools is not None else None
        self.tool_name_prefix = tool_name_prefix
        self.timeout_seconds = timeout_seconds
        self._mcp_tools: Any = None

    def status(self) -> Status:
        if self._mcp_tools is None:
            return Status(ok=True, detail=f"playwright-mcp ({self.browser}, not connected)")
        if getattr(self._mcp_tools, "initialized", False):
            return Status(ok=True, detail=f"playwright-mcp ({self.browser}, connected)")
        return Status(ok=True, detail=f"playwright-mcp ({self.browser}, pending)")

    async def astatus(self) -> Status:
        return self.status()

    def get_tools(self, *, exclude_interaction_tools: bool = False) -> list:
        if self._mcp_tools is None:
            self._mcp_tools = self._build_tools(exclude_interaction_tools=exclude_interaction_tools)
        return [self._mcp_tools]

    def _build_tools(self, *, exclude_interaction_tools: bool = False) -> Any:
        from mcp import StdioServerParameters

        from agno.tools.mcp import MCPTools

        cmd_args = [f"@playwright/mcp@{self.version}"]
        if self.headless:
            cmd_args.append("--headless")
        if self.browser != "chromium":
            cmd_args.append(f"--browser={self.browser}")

        server_params = StdioServerParameters(command="npx", args=cmd_args, env=self.env)

        # Merge user-provided exclude_tools with interaction tools if readonly
        exclude = set(self.exclude_tools) if self.exclude_tools else set()
        if exclude_interaction_tools:
            exclude.update(INTERACTION_TOOLS)

        return MCPTools(
            server_params=server_params,
            transport="stdio",
            include_tools=self.include_tools,
            exclude_tools=list(exclude) if exclude else None,
            tool_name_prefix=self.tool_name_prefix,
            timeout_seconds=self.timeout_seconds,
        )

    async def asetup(self, *, exclude_interaction_tools: bool = False) -> None:
        """Start the Playwright MCP server and connect.

        On failure, logs a warning; the browser backend will be
        unavailable until the next restart.
        """
        if self._mcp_tools is None:
            self._mcp_tools = self._build_tools(exclude_interaction_tools=exclude_interaction_tools)
        if getattr(self._mcp_tools, "initialized", False):
            return
        log_info(f"PlaywrightMCPBackend: starting npx @playwright/mcp@{self.version} ({self.browser})")
        try:
            await self._mcp_tools._connect()
        except Exception as exc:
            log_warning(f"PlaywrightMCPBackend setup failed — {type(exc).__name__}: {exc}")
            self._mcp_tools = None

    async def aclose(self) -> None:
        """Stop the MCP server and clear the cached tool handle."""
        tools = self._mcp_tools
        self._mcp_tools = None
        if tools is None:
            return
        try:
            await tools.close()
        except Exception as exc:
            log_warning(f"PlaywrightMCPBackend close raised {type(exc).__name__}: {exc}")
