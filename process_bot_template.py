import asyncio
import os

from browser_use import Agent
from browser_use.llm.google.chat import ChatGoogle


async def main() -> None:
    # Keep prompts short and explicit. Replace this with your new process.
    objective = (
        "Open https://example.com, sign in, navigate to the target page, "
        "and complete the first pending item."
    )

    llm = ChatGoogle(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )

    agent = Agent(
        task=objective,
        llm=llm,
        use_vision=True,
        directly_open_url=True,
        max_actions_per_step=6,
        initial_actions=[
            {"navigate": {"url": "https://example.com", "new_tab": False}}
        ],
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
