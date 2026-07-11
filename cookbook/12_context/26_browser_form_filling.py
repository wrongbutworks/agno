"""
Browser Context Provider - Form Filling
========================================

Demonstrates interactive browser automation: navigating to a page,
filling form fields, and submitting. The agent uses click/type tools
to interact with the page, not just read it.

Uses httpbin.org's form endpoint which echoes submitted data back,
letting us verify the form was filled correctly.

Requires:
    OPENAI_API_KEY
    Node.js 18+ (npx downloads @playwright/mcp on first run)
"""

import asyncio

from agno.agent import Agent
from agno.context.browser import BrowserContextProvider, PlaywrightMCPBackend
from agno.models.openai import OpenAIResponses


async def main() -> None:
    browser = BrowserContextProvider(
        backend=PlaywrightMCPBackend(headless=True),
        model=OpenAIResponses(id="gpt-5.5"),
    )

    await browser.asetup()
    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.5"),
            tools=browser.get_tools(),
            instructions=browser.instructions(),
            markdown=True,
        )

        prompt = (
            "Go to https://httpbin.org/forms/post and fill out the form with: "
            "Customer name: John Doe, "
            "Telephone: 555-1234, "
            "Email: john@example.com, "
            "Size: Medium, "
            "Topping: Bacon. "
            "Then submit the form and tell me what the response shows."
        )
        print(f"> {prompt}\n")
        await agent.aprint_response(prompt)
    finally:
        await browser.aclose()


if __name__ == "__main__":
    asyncio.run(main())
