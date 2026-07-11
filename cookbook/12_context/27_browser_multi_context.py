"""
Browser + Web Search - Multi-Context Research
==============================================

Combines BrowserContextProvider with WebContextProvider for research.
The agent can:
- Search the web via Exa to find relevant pages
- Browse to specific pages and interact with them

This pattern is useful when you need both discovery (web search) and
deep interaction (browser automation) in the same workflow.

Requires:
    OPENAI_API_KEY
    Node.js 18+ (npx downloads @playwright/mcp on first run)
    (optional) EXA_API_KEY for higher rate limits
"""

import asyncio

from agno.agent import Agent
from agno.context.browser import BrowserContextProvider, PlaywrightMCPBackend
from agno.context.web import ExaMCPBackend, WebContextProvider
from agno.models.openai import OpenAIResponses

provider_model = OpenAIResponses(id="gpt-5.5")

browser = BrowserContextProvider(
    backend=PlaywrightMCPBackend(headless=True),
    model=provider_model,
)

web = WebContextProvider(
    backend=ExaMCPBackend(),
    model=provider_model,
)


async def main() -> None:
    await browser.asetup()
    await web.asetup()
    try:
        tools = [*browser.get_tools(), *web.get_tools()]
        instructions = "\n".join([browser.instructions(), web.instructions()])

        agent = Agent(
            model=OpenAIResponses(id="gpt-5.5"),
            tools=tools,
            instructions=(
                "You have two context providers:\n"
                "- query_browser: for navigating to specific URLs and interacting with pages\n"
                "- query_web: for searching the web to discover relevant pages\n\n"
                + instructions
            ),
            markdown=True,
        )

        prompt = (
            "Search for the official Python documentation on asyncio, "
            "then go to the page and find what the main event loop methods are."
        )
        print(f"> {prompt}\n")
        await agent.aprint_response(prompt)
    finally:
        await browser.aclose()
        await web.aclose()


if __name__ == "__main__":
    asyncio.run(main())
