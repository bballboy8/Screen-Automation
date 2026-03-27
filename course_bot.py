import asyncio
import os

from browser_use import Agent
from browser_use.tools.service import Tools
from browser_use.llm.google.chat import ChatGoogle


def is_transient_browser_disconnect(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "websocket connection closed" in message
        or "target closed" in message
        or "session closed" in message
        or "connection closed" in message
    )


async def main():
    username = os.getenv("BOSTONIFI_USERNAME")
    password = os.getenv("BOSTONIFI_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Missing credentials. Set BOSTONIFI_USERNAME and BOSTONIFI_PASSWORD environment variables."
        )

    llm = ChatGoogle(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    fallback_llm = ChatGoogle(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )

    objective = (
        "Log in to https://learn.bostonifi.com/start "
        f"using username '{username}' and password '{password}'. "
        "\n\n"
        "NAVIGATION RULES — follow based on what page you are currently on:\n"
        "\n"
        "PAGE: My Courses list\n"
        "  - Scroll down to see all curricula.\n"
        "  - Click a curriculum accordion ONLY when it is currently collapsed (e.g., aria-expanded='false') to expand it.\n"
        "  - Never click an already expanded curriculum accordion header (aria-expanded='true'), because that collapses it.\n"
        "  - If you accidentally collapse a curriculum, immediately click it once to re-expand it before doing anything else.\n"
        "  - Click only real course/module title links or buttons (text like 'Course X - ...' or 'N. Topic Name').\n"
        "  - Never click decorative/status-only elements such as '*', bullets, icons, empty labels, progress badges, or generic container div/spans.\n"
        "  - If the candidate element text does not clearly include the expected course/module title, do not click it; refresh state and choose the explicit title element.\n"
        "  - Find the FIRST item that is NOT marked Passed, Completed, or 100% complete.\n"
        "  - Skip items labelled Supplemental Resource, QBank, Optional, or Practice.\n"
        "  - Click that item to open it. Do not click anything else on this page.\n"
        "\n"
        "PAGE: Course or module landing page (shows a Resume / Start / Continue button)\n"
        "  - Click exactly ONE of: 'Resume course', 'Start', 'Continue', or 'Let's continue'.\n"
        "  - Do not go back to My Courses from here.\n"
        "\n"
        "PAGE: Lesson / content screen (shows lesson text, video, or a Next / Complete and continue button)\n"
        "  - Click 'Complete and continue' or 'Next' to advance.\n"
        "  - Do not go back to My Courses from here.\n"
        "\n"
        "PAGE: Question / quiz screen (shows one or more questions with answer choices)\n"
        "  - Read every question carefully and answer every question shown on the current page before submitting.\n"
        "  - DO NOT pick default/first answers. Use reasoning to choose the best answer for each question.\n"
        "  - First, use evaluate() to extract a structured JSON snapshot of ALL visible questions, including question text and options.\n"
        "  - Then reason over that extracted content and choose answers before interacting with the page.\n"
        "  - IMPORTANT: Never rely on hardcoded or guessed question/fieldset IDs (for example ef-question-xxxxx from prior steps). IDs may change.\n"
        "  - Always locate questions by visible question text and option label text at interaction time.\n"
        "  - Use fuzzy text matching (case-insensitive contains) when matching option labels, because punctuation/spacing may differ.\n"
        "  - Support these types:\n"
        "    * Multiple choice/radio: choose exactly one option per question.\n"
        "    * Dropdown/select: choose the best option from each dropdown (never leave placeholder selected).\n"
        "    * Matching/drag-and-drop: map right-side choices to left-side prompts, then use evaluate() to dispatch dragstart/dragover/drop events to perform the match.\n"
        "  - RADIO RELIABILITY: many radios are custom-styled. If input.click() fails, click the associated label text/circle, then dispatch input+change events and verify checked=true.\n"
        "  - RADIO MATCHING: when selecting a radio answer, locate the target question block by question text, then locate the option label by text, then activate its linked input via label[for] or nearest input.\n"
        "  - If browser-level popups (password/save prompts) appear, dismiss them first (Escape) before trying to click answer controls again.\n"
        "  - Execution order: (1) extract page questions -> (2) decide answers -> (3) apply all answers with one robust evaluate() pass using text matching -> (4) verify selected states -> (5) submit.\n"
        "  - If any answer application fails, re-extract current visible DOM and retry once using text matching; do not repeat the same broken selector.\n"
        "  - If a click/select/drag action fails, refresh state once and retry with evaluate() using robust text-based selectors (not hardcoded question IDs).\n"
        "  - Confirm each question has a visible selected state (checked/highlighted/selected/matched) before clicking 'Submit'.\n"
        "  - Never click navigation tree items while unanswered questions remain on the page.\n"
        "  - After ALL questions on the page are answered, click 'Submit' once.\n"
        "  - After Submit, click 'Continue' or 'Next' to advance.\n"
        "  - Do not go back to My Courses from here.\n"
        "\n"
        "PAGE: Module complete / congratulations screen\n"
        "  - Click 'Continue' or 'Next' if present, otherwise navigate back to My Courses.\n"
        "\n"
        "GENERAL RULES:\n"
        "  - Never navigate away from a module mid-lesson. Stay in the module until it shows complete.\n"
        "  - Never click the browser Back button inside a module.\n"
        "  - Never click elements whose visible text is only '*' or other punctuation.\n"
        "  - Prefer clicking anchor/button elements with full human-readable titles over parent containers.\n"
        "  - Never call done() until you have expanded every curriculum on My Courses and verified every required item is complete.\n"
        "  - After every click, wait for the page to finish loading before clicking again.\n"
    )

    constrained_tools = Tools(exclude_actions=["find_elements", "write_file", "replace_file", "read_file"])

    def build_agent() -> Agent:
        return Agent(
            task=objective,
            llm=llm,
            fallback_llm=fallback_llm,
            tools=constrained_tools,
            include_attributes=["title", "aria-label", "id", "name", "role", "value", "checked", "aria-checked"],
            use_vision=True,
            directly_open_url=True,
            use_thinking=False,
            enable_planning=False,
            max_actions_per_step=2,
            max_failures=20,
            final_response_after_failure=False,
            use_judge=False,
            available_file_paths=[],
            max_clickable_elements_length=120000,
            initial_actions=[
                {"navigate": {"url": "https://learn.bostonifi.com/start", "new_tab": False}}
            ],
        )

    last_error = None
    for attempt in range(1, 4):
        agent = build_agent()
        try:
            await agent.run()
            return
        except Exception as error:
            if not is_transient_browser_disconnect(error) or attempt == 3:
                raise
            last_error = error
            print(f"Transient browser disconnect on attempt {attempt}; restarting agent...")
            await asyncio.sleep(3)

    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    asyncio.run(main())
