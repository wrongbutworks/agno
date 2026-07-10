"""Unit tests for BrowserContextProvider."""

from __future__ import annotations

import pytest

from agno.context.browser import BrowserContextProvider, PlaywrightMCPBackend
from agno.context.browser.playwright_mcp import INTERACTION_TOOLS
from agno.context.mode import ContextMode


class TestPlaywrightMCPBackend:
    def test_default_status_not_connected(self):
        backend = PlaywrightMCPBackend()
        status = backend.status()
        assert status.ok is True
        assert "not connected" in status.detail
        assert "chromium" in status.detail

    def test_default_browser_is_chromium(self):
        backend = PlaywrightMCPBackend()
        assert backend.browser == "chromium"

    def test_custom_browser(self):
        backend = PlaywrightMCPBackend(browser="firefox")
        assert backend.browser == "firefox"
        assert "firefox" in backend.status().detail

    def test_headless_default_true(self):
        backend = PlaywrightMCPBackend()
        assert backend.headless is True

    def test_headless_false(self):
        backend = PlaywrightMCPBackend(headless=False)
        assert backend.headless is False

    def test_default_version_is_latest(self):
        backend = PlaywrightMCPBackend()
        assert backend.version == "latest"

    def test_custom_version(self):
        backend = PlaywrightMCPBackend(version="1.2.3")
        assert backend.version == "1.2.3"

    def test_default_include_tools_is_none(self):
        backend = PlaywrightMCPBackend()
        assert backend.include_tools is None

    def test_custom_include_tools(self):
        backend = PlaywrightMCPBackend(include_tools=["browser_navigate", "browser_snapshot"])
        assert backend.include_tools == ["browser_navigate", "browser_snapshot"]

    def test_tool_name_prefix(self):
        backend = PlaywrightMCPBackend(tool_name_prefix="pw_")
        assert backend.tool_name_prefix == "pw_"

    def test_env_parameter(self):
        backend = PlaywrightMCPBackend(env={"HTTP_PROXY": "http://proxy:8080"})
        assert backend.env == {"HTTP_PROXY": "http://proxy:8080"}

    def test_interaction_tools_constant_exists(self):
        assert "browser_click" in INTERACTION_TOOLS
        assert "browser_type" in INTERACTION_TOOLS
        assert "browser_evaluate" in INTERACTION_TOOLS


class TestBrowserContextProvider:
    def test_default_is_read_only(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend)
        assert provider.read is True
        assert provider.write is False

    def test_tool_names(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend)
        assert provider.query_tool_name == "query_browser"
        assert provider.update_tool_name == "update_browser"

    def test_custom_id_changes_tool_names(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, id="chrome")
        assert provider.query_tool_name == "query_chrome"
        assert provider.update_tool_name == "update_chrome"

    def test_status_delegates_to_backend(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend)
        status = provider.status()
        assert status.ok is True
        assert "playwright-mcp" in status.detail

    def test_instructions_read_only_mode(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=False)
        instructions = provider.instructions()
        assert "query_browser" in instructions
        assert "update_browser" not in instructions

    def test_instructions_read_write_mode(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=True)
        instructions = provider.instructions()
        assert "query_browser" in instructions
        assert "update_browser" in instructions

    def test_instructions_tools_mode_readonly(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, mode=ContextMode.tools, write=False)
        instructions = provider.instructions()
        assert "read-only" in instructions

    def test_instructions_tools_mode_write(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, mode=ContextMode.tools, write=True)
        instructions = provider.instructions()
        assert "interaction" in instructions

    def test_default_tools_returns_query_tool_only_when_read_only(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=False)
        tools = provider.get_tools()
        tool_names = [t.name for t in tools]
        assert tool_names == ["query_browser"]

    def test_default_tools_returns_both_when_write_enabled(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=True)
        tools = provider.get_tools()
        tool_names = [t.name for t in tools]
        assert tool_names == ["query_browser", "update_browser"]

    def test_all_tools_mode_returns_backend_tools(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, mode=ContextMode.tools)
        tools = provider.get_tools()
        assert len(tools) == 1

    def test_sync_query_raises_not_implemented(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend)
        with pytest.raises(NotImplementedError, match="async-only"):
            provider.query("search something")

    def test_sync_update_raises_not_implemented(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=True)
        with pytest.raises(NotImplementedError, match="async-only"):
            provider.update("click something")

    @pytest.mark.asyncio
    async def test_aupdate_raises_when_read_only(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=False)
        with pytest.raises(NotImplementedError, match="read-only"):
            await provider.aupdate("click something")

    @pytest.mark.asyncio
    async def test_aget_update_agent_raises_when_read_only(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=False)
        with pytest.raises(NotImplementedError, match="read-only"):
            await provider._aget_update_agent(None)

    def test_custom_tool_names(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(
            backend=backend,
            query_tool_name="search_web",
            update_tool_name="interact_web",
        )
        assert provider.query_tool_name == "search_web"
        assert provider.update_tool_name == "interact_web"

    def test_stream_sub_agent_events_forwarded(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, stream_sub_agent_events=False)
        assert provider.stream_sub_agent_events is False

    @pytest.mark.asyncio
    async def test_aclose_clears_agent_cache(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend)
        _ = provider._ensure_read_agent()
        assert provider._read_agent is not None
        await provider.aclose()
        assert provider._read_agent is None

    def test_separate_read_write_agents(self):
        backend = PlaywrightMCPBackend()
        provider = BrowserContextProvider(backend=backend, write=True)
        read_agent = provider._ensure_read_agent()
        write_agent = provider._ensure_write_agent()
        assert read_agent is not write_agent
        assert read_agent.id == "browser-read"
        assert write_agent.id == "browser-write"
