"""
Browser Context Provider
========================

Browser automation via a configurable backend. Two tools:

- ``query_<id>`` — natural-language reads (navigate, snapshot, extract
  content from web pages). Interaction tools are excluded.
- ``update_<id>`` — natural-language writes (click, type, submit forms).
  Disabled by default (``write=False``).

The default backend is ``PlaywrightMCPBackend``, which runs Playwright's
official MCP server via stdio. Uses the accessibility tree by default,
which is ~4x more token-efficient than vision-based approaches.

Read/write enforcement: when ``write=False``, interaction tools
(click, type, fill_form, evaluate, etc.) are excluded from the sub-agent's
toolkit via the backend's ``exclude_interaction_tools`` flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agno.agent import Agent
from agno.context._utils import answer_from_run
from agno.context.backend import ContextBackend
from agno.context.mode import ContextMode
from agno.context.provider import Answer, ContextProvider, Status
from agno.run import RunContext

if TYPE_CHECKING:
    from agno.models.base import Model


class BrowserContextProvider(ContextProvider):
    """Browser automation via query/update tools, backed by Playwright MCP."""

    def __init__(
        self,
        backend: ContextBackend,
        *,
        id: str = "browser",
        name: str = "Browser",
        instructions: str | None = None,
        mode: ContextMode = ContextMode.default,
        model: Model | None = None,
        read: bool = True,
        write: bool = False,
        query_tool_name: str | None = None,
        update_tool_name: str | None = None,
        stream_sub_agent_events: bool = True,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            mode=mode,
            model=model,
            read=read,
            write=write,
            query_tool_name=query_tool_name,
            update_tool_name=update_tool_name,
            stream_sub_agent_events=stream_sub_agent_events,
        )
        self.backend = backend
        self.custom_instructions = instructions
        self._read_agent: Agent | None = None
        self._write_agent: Agent | None = None

    def status(self) -> Status:
        return self.backend.status()

    async def astatus(self) -> Status:
        return await self.backend.astatus()

    async def asetup(self) -> None:
        exclude = not self.write
        try:
            await self.backend.asetup(exclude_interaction_tools=exclude)  # type: ignore[call-arg]
        except TypeError:
            await self.backend.asetup()

    async def aclose(self) -> None:
        self._read_agent = None
        self._write_agent = None
        await self.backend.aclose()

    def query(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        raise NotImplementedError(
            "BrowserContextProvider does not support sync query(); use aquery() (MCP sessions are async-only)."
        )

    async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        agent = self._ensure_read_agent()
        kwargs = self._run_kwargs_for_sub_agent(run_context)
        return answer_from_run(await agent.arun(question, **kwargs))

    def update(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        raise NotImplementedError(
            "BrowserContextProvider does not support sync update(); use aupdate() (MCP sessions are async-only)."
        )

    async def aupdate(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        if not self.write:
            raise NotImplementedError(f"{self.name} is read-only. Set write=True to enable interactions.")
        agent = self._ensure_write_agent()
        kwargs = self._run_kwargs_for_sub_agent(run_context)
        return answer_from_run(await agent.arun(instruction, **kwargs))

    def instructions(self) -> str:
        if self.mode == ContextMode.tools:
            if self.write:
                return f"`{self.name}`: browser tools for navigation, snapshots, screenshots, and interaction."
            return f"`{self.name}`: browser tools for navigation, snapshots, and screenshots (read-only)."
        if self.write:
            return (
                f"`{self.name}`: call `{self.query_tool_name}(question)` to browse and find information. "
                f"Use `{self.update_tool_name}(instruction)` to interact (click, type, submit)."
            )
        return (
            f"`{self.name}`: call `{self.query_tool_name}(question)` to browse the web, "
            "navigate pages, and extract information."
        )

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def _default_tools(self) -> list:
        return self._read_write_tools()

    def _all_tools(self) -> list:
        exclude = not self.write
        try:
            return self.backend.get_tools(exclude_interaction_tools=exclude)  # type: ignore[call-arg]
        except TypeError:
            return self.backend.get_tools()

    # ------------------------------------------------------------------
    # Sub-agents
    # ------------------------------------------------------------------

    async def _aget_query_agent(self, run_context):
        return self._ensure_read_agent()

    async def _aget_update_agent(self, run_context):
        if not self.write:
            raise NotImplementedError(f"{self.name} is read-only. Set write=True to enable interactions.")
        return self._ensure_write_agent()

    def _ensure_read_agent(self) -> Agent:
        if self._read_agent is None:
            self._read_agent = self._build_read_agent()
        return self._read_agent

    def _ensure_write_agent(self) -> Agent:
        if self._write_agent is None:
            self._write_agent = self._build_write_agent()
        return self._write_agent

    def _build_read_agent(self) -> Agent:
        try:
            tools = self.backend.get_tools(exclude_interaction_tools=True)  # type: ignore[call-arg]
        except TypeError:
            tools = self.backend.get_tools()
        return Agent(
            id=f"{self.id}-read",
            name=f"{self.name} Read",
            model=self.model,
            instructions=self.custom_instructions or DEFAULT_READ_INSTRUCTIONS,
            tools=tools,
            markdown=True,
        )

    def _build_write_agent(self) -> Agent:
        return Agent(
            id=f"{self.id}-write",
            name=f"{self.name} Write",
            model=self.model,
            instructions=self.custom_instructions or DEFAULT_WRITE_INSTRUCTIONS,
            tools=self.backend.get_tools(),
            markdown=True,
        )


DEFAULT_READ_INSTRUCTIONS = """\
You browse the web to find information. You are READ-ONLY.

## Workflow

1. **Navigate first.** Use `browser_navigate` to go to a URL.

2. **Take a snapshot.** Use `browser_snapshot` to get the page's accessibility
   tree. This shows interactive elements with their targets.

3. **Use screenshots sparingly.** Only use `browser_take_screenshot` when you
   need visual layout, images, or content not in the accessibility tree.

4. **Extract information.** Read the snapshot to find what you need. Quote
   relevant text verbatim. Include URLs for pages you visit.

5. **Navigate via URL.** Use `browser_navigate` for new pages. You cannot
   click links — use the URL directly.

## Safety

- You are read-only. You cannot click, type, or submit forms.
- If you need to interact with a page, say so and stop.
"""


DEFAULT_WRITE_INSTRUCTIONS = """\
You browse the web and interact with pages.

## Workflow

1. **Navigate first.** Use `browser_navigate` to go to a URL.

2. **Take a snapshot.** Use `browser_snapshot` to get the page's accessibility
   tree. Elements have `target` attributes for interaction.

3. **Interact with elements.** Use the target from the snapshot:
   - `browser_click(target="Submit button")` — click an element
   - `browser_type(target="Search input", text="query")` — type text

4. **Use screenshots sparingly.** Only use `browser_take_screenshot` when you
   need visual layout not captured in the snapshot.

## Safety

- You are operating a real browser. Actions affect real websites.
- Never submit forms with sensitive data unless explicitly instructed.
- Never authenticate or enter credentials.
- If a page asks for login, report it and stop.
"""
