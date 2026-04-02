import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field

from browser_use import Agent
from browser_use.llm.messages import SystemMessage, UserMessage
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


@dataclass
class ProgressGuardState:
    last_url: str = ""
    same_url_steps: int = 0
    consecutive_skeleton_steps: int = 0
    forced_reload_count: int = 0
    last_forced_click_ts: float = 0.0
    last_quiz_signature: str = ""
    last_quiz_action_ts: float = 0.0
    last_forced_home_redirect_ts: float = 0.0
    last_completion_redirect_ts: float = 0.0
    quiz_attempts: dict = field(default_factory=dict)  # quiz_signature -> list[list[str]] of tried input_id sets


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
        model="gemini-2.5-pro",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    quiz_reasoner_llm = ChatGoogle(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    quiz_reasoner_fallback_llm = ChatGoogle(
        model="gemini-2.5-pro",
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
        "  - Read every question carefully and answer every question shown before submitting.\n"
        "  - DO NOT pick default/first answers. Use CFP exam reasoning to choose the BEST answer for each question.\n"
        "  - Quiz Handler extracts all questions automatically. DO NOT attempt to re-extract with evaluate().\n"
        "  - Use the extracted questions already provided in logs/context, reason with CFP expertise, and select the best answer(s).\n"
        "  - Verify all questions show selected state, then click 'Submit'.\n"
        "  - After Submit, click 'Continue' or 'Next' to advance. Do NOT go back to My Courses.\n"
        "\n"
        "PAGE: Module complete / congratulations screen\n"
        "  - Click only in-page progression controls such as 'Continue', 'Next', or 'Repeat test'.\n"
        "  - If no progression control exists, wait and re-check; do NOT navigate back to My Courses.\n"
        "\n"
        "GENERAL RULES:\n"
        "  - Never navigate away from a module mid-lesson. Stay in the module until it shows complete.\n"
        "  - Never click the browser Back button inside a module.\n"
        "  - Never click elements whose visible text is only '*' or other punctuation.\n"
        "  - Prefer clicking anchor/button elements with full human-readable titles over parent containers.\n"
        "  - Never call done() until you have expanded every curriculum on My Courses and verified every required item is complete.\n"
        "  - After every click, wait for the page to finish loading before clicking again.\n"
        "  - If the page shows loading skeletons/placeholders, wait for real content; do not navigate away just because loading is slow.\n"
        "  - Do not leave a module for 'My Courses' unless the module is clearly completed (explicit completion/result screen).\n"
        "\n"
        "STRICTLY FORBIDDEN ACTIONS — violating these will break the automation:\n"
        "  - NEVER click any element with role='treeitem' (the sidebar lesson navigation panel). These are disabled and clicks will fail.\n"
        "  - NEVER click any 'My Courses' breadcrumb, header link, or top navigation link at any time after login. Those elements are disabled.\n"
        "  - NEVER use the navigate action to go to any URL after initial startup. URL-level navigation is blocked.\n"
        "  - NEVER use the go_back action.\n"
        "  - NEVER click 'Previous unit', 'Previous lesson', or any backward section navigation control.\n"
        "  - The ONLY valid navigation actions are in-page buttons: 'Complete and continue', 'Next', 'Submit', 'Repeat test', 'Resume course', 'Start', \"Let's continue\", or course/module title links on the My Courses list itself.\n"
        "  - If 'Complete and continue' does not seem to advance the page, wait 3 seconds and click it again — do NOT switch to tree navigation or go back to My Courses.\n"
    )

    constrained_tools = Tools(exclude_actions=["find_elements", "write_file", "replace_file", "read_file", "extract", "go_back", "navigate"])

    guard = ProgressGuardState()

    async def enforce_allowed_domain(agent: Agent) -> None:
        """Keep the agent on BostonIFI pages only.
        If startup or a model action lands on a search engine/new tab, force back to the target homepage."""
        page = await agent.browser_session.get_current_page()
        if page is None:
            return

        try:
            current_url = await agent.browser_session.get_current_page_url()
        except Exception:
            return

        if not current_url:
            return

        allowed_host_fragment = "learn.bostonifi.com"
        if allowed_host_fragment in current_url.lower():
            return

        now = time.time()
        if (now - guard.last_forced_home_redirect_ts) < 2.0:
            return

        print(f"[Domain Guard] Unexpected URL '{current_url}'. Redirecting to BostonIFI homepage...")
        try:
            # browser-use page wrapper does not accept Playwright's wait_until kwarg
            await page.goto("https://learn.bostonifi.com/start")
        except Exception:
            # Fallback: hard client-side redirect if goto wrapper changes/errs
            await page.evaluate("""() => { window.location.href = 'https://learn.bostonifi.com/start'; }""")
        guard.last_forced_home_redirect_ts = now
        await asyncio.sleep(1.5)

    async def disable_forbidden_nav(agent: Agent) -> None:
        """Before each step: neutralise sidebar treeitems, My Courses breadcrumb,
        and decorative/punctuation-only links so the LLM cannot pick them."""
        page = await agent.browser_session.get_current_page()
        if page is None:
            return
        try:
            await page.evaluate(
                """() => {
                    // Sidebar lesson navigation — agent must never click individual lesson nodes
                    document.querySelectorAll('[role="treeitem"]').forEach(el => {
                        el.style.pointerEvents = 'none';
                        el.style.cursor = 'default';
                        el.setAttribute('aria-disabled', 'true');
                        el.setAttribute('tabindex', '-1');
                    });

                    // Top breadcrumb / header "My Courses" link
                    document.querySelectorAll('a, button, span').forEach(el => {
                        const txt = (el.textContent || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                        if (txt.includes('my courses')) {
                            const clickableParent = el.closest('a, button') || el;
                            clickableParent.style.pointerEvents = 'none';
                            clickableParent.style.display = 'none';
                            clickableParent.setAttribute('aria-disabled', 'true');
                            clickableParent.setAttribute('tabindex', '-1');
                            if (clickableParent.tagName === 'A') {
                                clickableParent.removeAttribute('href');
                            }
                        }
                    });

                    // Any explicit links/buttons that route back to course list
                    document.querySelectorAll('a, button').forEach(el => {
                        const txt = (el.textContent || el.getAttribute('value') || '').trim().toLowerCase();
                        const href = (el.getAttribute('href') || '').toLowerCase();
                        const routesToCourses = href.includes('/start') || href.includes('/my-courses') || href.includes('/my_courses');
                        const looksLikeBackNav = txt.includes('my courses') || txt === 'back to courses' || txt === 'back';
                        if (routesToCourses || looksLikeBackNav) {
                            el.style.pointerEvents = 'none';
                            el.style.display = 'none';
                            el.setAttribute('aria-disabled', 'true');
                            el.setAttribute('tabindex', '-1');
                            if (el.tagName === 'A') {
                                el.removeAttribute('href');
                            }
                        }
                    });

                    // Disable any unit/lesson backward navigation links (e.g. id=ef-navigate-previous)
                    document.querySelectorAll('a, button, [id], [data-action]').forEach(el => {
                        const id = (el.id || '').toLowerCase();
                        const action = (el.getAttribute('data-action') || '').toLowerCase();
                        const txt = (el.textContent || el.getAttribute('value') || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                        const href = (el.getAttribute('href') || '').toLowerCase();

                        const isBackwardNav =
                            id.includes('navigate-previous') ||
                            id.includes('previous-unit') ||
                            action.includes('previous') ||
                            txt.startsWith('previous unit') ||
                            txt.startsWith('previous lesson') ||
                            txt === 'previous' ||
                            href.includes('navigate-previous');

                        if (isBackwardNav) {
                            const clickableParent = el.closest('a, button') || el;
                            clickableParent.style.pointerEvents = 'none';
                            clickableParent.style.display = 'none';
                            clickableParent.setAttribute('aria-disabled', 'true');
                            clickableParent.setAttribute('tabindex', '-1');
                            if (clickableParent.tagName === 'A') {
                                clickableParent.removeAttribute('href');
                            }
                        }
                    });

                    // Decorative / punctuation-only links (e.g. "*", "•", single chars)
                    // that appear as status icons inside course accordion lists
                    document.querySelectorAll('a').forEach(el => {
                        const txt = (el.textContent || '').trim();
                        const hasRealText = txt.length > 3 && /[a-zA-Z0-9]/.test(txt);
                        if (!hasRealText) {
                            el.style.pointerEvents = 'none';
                            el.setAttribute('aria-disabled', 'true');
                        }
                    });
                }"""
            )
        except Exception:
            pass

    async def auto_handle_quiz(agent: Agent) -> None:
        """Auto-detect quiz pages, extract questions, ask LLM for choices, click answers, and submit."""
        page = await agent.browser_session.get_current_page()
        if page is None:
            return

        try:
            # Detect if we're on a quiz page
            quiz_check = await page.evaluate(
                """() => {
                    const hasQuestions = document.querySelector('fieldset[id^="ef-question-"]') !== null;
                    const hasSubmit = document.querySelector('input[type="submit"]') !== null;
                    return { hasQuestions, hasSubmit };
                }"""
            )
            quiz_info = json.loads(quiz_check)
            if not quiz_info.get("hasQuestions"):
                return

            print("[Quiz Handler] Detected quiz page; extracting questions with proven code...")

            # Extract ALL questions using robust, tested DOM traversal
            extraction_result = await page.evaluate(
                """() => {
                    const questions = [];
                    document.querySelectorAll('fieldset[id^="ef-question-"]').forEach((fieldset, idx) => {
                        const container =
                            fieldset.closest('.que, .question, [id^="question-"], [class*="question"]') ||
                            fieldset.parentElement ||
                            fieldset;
                        const legendText = (fieldset.querySelector('legend')?.textContent || '').trim();

                        const options = [];
                        fieldset.querySelectorAll('input[type="radio"], input[type="checkbox"]').forEach(input => {
                            const label = input.parentElement?.tagName === 'LABEL' ? input.parentElement : document.querySelector(`label[for="${input.id}"]`);
                            const text = (label ? label.textContent : input.value || '').replace(/\\s+/g, ' ').trim();
                            options.push({
                                text: text.substring(0, 150),
                                inputId: input.id,
                                inputName: input.name,
                                inputValue: input.value,
                                inputType: input.type,
                                disabled: input.disabled
                            });
                        });

                        const optionTexts = options.map(o => o.text.toLowerCase()).filter(Boolean);
                        let questionText = '';

                        const stemSelectors = [
                            '.qtext',
                            '.questiontext',
                            '.prompt',
                            '.stem',
                            '.question-content p',
                            'p'
                        ];
                        for (const sel of stemSelectors) {
                            const candidates = Array.from(container.querySelectorAll(sel));
                            for (const el of candidates) {
                                const txt = (el.textContent || '').replace(/\\s+/g, ' ').trim();
                                if (!txt || txt.length < 12) continue;
                                const txtLower = txt.toLowerCase();
                                if (txtLower === legendText.toLowerCase()) continue;
                                if (optionTexts.includes(txtLower)) continue;
                                if (/^question\\s*\\d+/i.test(txt) && txt.length < 40) continue;
                                questionText = txt.substring(0, 300);
                                break;
                            }
                            if (questionText) break;
                        }

                        // Fallback: inspect direct text nodes in fieldset that are not option text.
                        if (!questionText) {
                            for (let node of fieldset.childNodes) {
                                if (node.nodeType !== Node.TEXT_NODE) continue;
                                const txt = (node.textContent || '').replace(/\\s+/g, ' ').trim();
                                if (!txt || txt.length < 12) continue;
                                const txtLower = txt.toLowerCase();
                                if (txtLower === legendText.toLowerCase()) continue;
                                if (optionTexts.includes(txtLower)) continue;
                                if (/^question\\s*\\d+/i.test(txt) && txt.length < 40) continue;
                                questionText = txt.substring(0, 300);
                                break;
                            }
                        }

                        // Last fallback: best-effort use legend if no stem found.
                        if (!questionText && legendText) {
                            questionText = legendText.substring(0, 300);
                        }

                        if (questionText || options.length > 0) {
                            questions.push({
                                questionNumber: idx + 1,
                                questionText: questionText || `Question ${idx + 1}`,
                                legendText,
                                options: options
                            });
                        }
                    });
                    return JSON.stringify(questions);
                }"""
            )

            try:
                questions = json.loads(extraction_result)
            except json.JSONDecodeError:
                print("[Quiz Handler] Extraction failed; skipping auto-quiz.")
                return

            if not questions:
                print("[Quiz Handler] No questions extracted.")
                return

            print(f"[Quiz Handler] Extracted {len(questions)} questions. Passing to LLM for reasoning...")
            
            # Log extracted questions for LLM context
            print("\n[Quiz Questions for LLM Reasoning]:")
            for q in questions:
                print(f"Q{q['questionNumber']}: {q['questionText']}")
                for i, opt in enumerate(q['options'], 1):
                    print(f"  {i}. {opt['text']}")

            # Skip repeated re-answering if this exact quiz payload was just processed.
            quiz_signature = json.dumps(questions, sort_keys=True)
            now = time.time()
            previous_attempts = guard.quiz_attempts.get(quiz_signature, [])
            attempt_number = len(previous_attempts) + 1
            # Use a shorter cooldown when retrying so we can try different answers sooner.
            cooldown = 3.0 if previous_attempts else 12.0
            if guard.last_quiz_signature == quiz_signature and (now - guard.last_quiz_action_ts) < cooldown:
                return
            if previous_attempts:
                print(f"[Quiz Handler] Retrying quiz (attempt #{attempt_number}); {len(previous_attempts)} previous incorrect attempt(s) will be excluded from LLM choices.")

            def parse_answer_payload(raw_text: str) -> dict | None:
                if not raw_text:
                    return None
                text = raw_text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?", "", text).strip()
                    text = re.sub(r"```$", "", text).strip()

                try:
                    parsed = json.loads(text)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass

                # Try to salvage the largest JSON object in the response.
                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    candidate = match.group(0)
                    try:
                        parsed = json.loads(candidate)
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        return None
                return None

            async def request_answers_for_batch(batch_questions: list[dict], batch_label: str) -> dict | None:
                lines = []
                for q in batch_questions:
                    lines.append(f"Q{q['questionNumber']}: {q['questionText']}")
                    for idx, opt in enumerate(q.get("options", []), 1):
                        lines.append(f"  {idx}. {opt.get('text', '')} [id={opt.get('inputId', '')}]")

                # Build an id->readable-text map for options in this batch
                id_to_label: dict[str, str] = {}
                for q in batch_questions:
                    for opt in q.get("options", []):
                        iid = opt.get("inputId", "")
                        if iid:
                            id_to_label[iid] = f"Q{q['questionNumber']}: {opt.get('text', '')}"

                # Build retry context from prior failed answer sets
                retry_context = ""
                if previous_attempts:
                    retry_lines = []
                    for attempt_idx, prev_ids in enumerate(previous_attempts[-5:], 1):
                        relevant = [id_to_label[iid] for iid in prev_ids if iid in id_to_label]
                        if relevant:
                            retry_lines.append(f"  Attempt {attempt_idx}: {'; '.join(relevant)}")
                    if retry_lines:
                        retry_context = (
                            "\n\nCRITICAL: The following previous answer attempt(s) were WRONG. "
                            "Do NOT repeat these selections. Reason carefully and choose a DIFFERENT combination:\n"
                            + "\n".join(retry_lines) + "\n\n"
                        )

                llm_prompt = (
                    "You are a CFP exam expert. Choose the best answer for each question. "
                    "Return ONLY valid JSON with this schema: "
                    "{\"answers\": [{\"questionNumber\": <int>, \"optionIndexes\": [<int>, ...], \"inputIds\": [<string>, ...]}]}. "
                    "ALWAYS include inputIds using the exact ids shown beside options. Include optionIndexes as backup. "
                    "IMPORTANT: optionIndexes are 1-based (first option is 1). "
                    "For single-choice/radio questions choose EXACTLY ONE answer. "
                    "For checkbox questions include all correct options.\n\n"
                    + retry_context
                    + "Questions:\n"
                    + "\n".join(lines)
                )

                for attempt in range(1, 4):
                    raw_text = ""
                    try:
                        completion = await quiz_reasoner_llm.ainvoke(
                            [
                                SystemMessage(content="Return strict JSON only. No markdown."),
                                UserMessage(content=llm_prompt),
                            ],
                        )
                        raw_text = str(completion.completion or "").strip()
                    except Exception:
                        try:
                            completion = await quiz_reasoner_fallback_llm.ainvoke(
                                [
                                    SystemMessage(content="Return strict JSON only. No markdown."),
                                    UserMessage(content=llm_prompt),
                                ],
                            )
                            raw_text = str(completion.completion or "").strip()
                        except Exception as llm_err:
                            print(f"[Quiz Handler] {batch_label} LLM attempt {attempt} failed: {llm_err}")
                            await asyncio.sleep(1.2)
                            continue

                    print(f"[Quiz Handler] {batch_label} raw answer payload: {raw_text[:800]}")
                    parsed = parse_answer_payload(raw_text)
                    if isinstance(parsed, dict) and isinstance(parsed.get("answers"), list):
                        return parsed

                    print(f"[Quiz Handler] {batch_label} parse failed on attempt {attempt}; retrying...")
                    await asyncio.sleep(0.8)

                # If this batch fails repeatedly, split into smaller batches to avoid token/format issues.
                if len(batch_questions) > 1:
                    mid = len(batch_questions) // 2
                    left = await request_answers_for_batch(batch_questions[:mid], f"{batch_label}.A")
                    right = await request_answers_for_batch(batch_questions[mid:], f"{batch_label}.B")

                    merged_answers: list[dict] = []
                    if isinstance(left, dict) and isinstance(left.get("answers"), list):
                        merged_answers.extend([a for a in left["answers"] if isinstance(a, dict)])
                    if isinstance(right, dict) and isinstance(right.get("answers"), list):
                        merged_answers.extend([a for a in right["answers"] if isinstance(a, dict)])

                    if merged_answers:
                        print(f"[Quiz Handler] {batch_label} recovered via sub-batching ({len(merged_answers)} answers).")
                        return {"answers": merged_answers}

                return None

            # Large pages can exceed model output limits; answer in deterministic batches.
            # User preference: answer one question at a time for reliability on long exams.
            batch_size = 1
            all_answers: list[dict] = []
            failed_batches: list[str] = []
            for start in range(0, len(questions), batch_size):
                batch_questions = questions[start:start + batch_size]
                batch_number = (start // batch_size) + 1
                batch_label = f"Batch {batch_number}/{(len(questions) + batch_size - 1) // batch_size}"
                payload = await request_answers_for_batch(batch_questions, batch_label)
                if not payload:
                    print(f"[Quiz Handler] {batch_label} could not produce valid JSON answers.")
                    failed_batches.append(batch_label)
                    continue
                all_answers.extend([a for a in payload.get("answers", []) if isinstance(a, dict)])

            if failed_batches:
                print(f"[Quiz Handler] Skipping submit because some questions failed LLM answering: {', '.join(failed_batches[:20])}")
                return

            answer_payload = {"answers": all_answers}

            selected_input_ids: list[str] = []
            selected_human_lines: list[str] = []
            for answer in answer_payload["answers"]:
                if not isinstance(answer, dict):
                    continue
                qn = answer.get("questionNumber")
                option_indexes = answer.get("optionIndexes", [])
                explicit_ids = answer.get("inputIds", [])
                if not isinstance(qn, int):
                    continue
                question = next((q for q in questions if q.get("questionNumber") == qn), None)
                if not question:
                    continue
                options = question.get("options", [])
                if not options:
                    continue

                is_checkbox_question = any((opt.get("inputType") == "checkbox") for opt in options)
                candidate_ids: list[str] = []
                explicit_candidate_ids: list[str] = []

                if isinstance(explicit_ids, list):
                    for candidate_id in explicit_ids:
                        if isinstance(candidate_id, str) and candidate_id:
                            opt = next((o for o in options if o.get("inputId") == candidate_id), None)
                            if opt:
                                explicit_candidate_ids.append(candidate_id)

                if not isinstance(option_indexes, list):
                    option_indexes = []

                has_zero_index = any(isinstance(i, int) and i == 0 for i in option_indexes)
                for raw_idx in option_indexes:
                    if not isinstance(raw_idx, int):
                        continue

                    # If any 0 is present, treat whole set as 0-based and shift.
                    normalized_idx = (raw_idx + 1) if has_zero_index else raw_idx

                    if 1 <= normalized_idx <= len(options):
                        chosen_opt = options[normalized_idx - 1]
                        input_id = chosen_opt.get("inputId")
                        if isinstance(input_id, str) and input_id:
                            candidate_ids.append(input_id)

                # Prefer explicit ids from LLM; use index-derived ids only as fallback.
                if explicit_candidate_ids:
                    candidate_ids = explicit_candidate_ids + candidate_ids

                # De-duplicate while preserving order.
                dedup_candidate_ids: list[str] = []
                for cid in candidate_ids:
                    if cid not in dedup_candidate_ids:
                        dedup_candidate_ids.append(cid)

                if not dedup_candidate_ids:
                    continue

                if is_checkbox_question:
                    chosen_ids_for_question = dedup_candidate_ids
                else:
                    # Radio question: choose exactly one answer to avoid contradictory selections.
                    chosen_ids_for_question = [dedup_candidate_ids[0]]

                for chosen_id in chosen_ids_for_question:
                    selected_input_ids.append(chosen_id)
                    opt = next((o for o in options if o.get("inputId") == chosen_id), None)
                    if opt:
                        selected_human_lines.append(f"Q{qn}: {opt.get('text', '')}")

            if not selected_input_ids:
                print("[Quiz Handler] LLM returned no selectable answers.")
                return

            if selected_human_lines:
                print("[Quiz Handler] LLM selected options:")
                for line in selected_human_lines[:30]:
                    print(f"  - {line}")

            ids_json = json.dumps(list(dict.fromkeys(selected_input_ids)))
            apply_result = await page.evaluate(
                f"""() => {{
                    const selectedIds = {ids_json};
                    let clicked = 0;

                    selectedIds.forEach((id) => {{
                        const input = document.getElementById(id);
                        if (!input || input.disabled) return;

                        const isCheckbox = input.type === 'checkbox';
                        const isRadio = input.type === 'radio';

                        if (isRadio && input.name) {{
                            // Use click for radio so framework bindings update consistently.
                            if (!input.checked) input.click();
                        }} else if (isCheckbox) {{
                            // For checkbox multi-select, only click when unchecked to avoid toggling off.
                            if (!input.checked) input.click();
                        }} else {{
                            if (!input.checked) input.click();
                        }}

                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        clicked += 1;
                    }});

                    // Verify selections persisted; recover missing ones without label toggles.
                    const missing = [];
                    selectedIds.forEach((id) => {{
                        const input = document.getElementById(id);
                        if (!input || input.disabled) return;
                        if (!input.checked) missing.push(id);
                    }});

                    missing.forEach((id) => {{
                        const input = document.getElementById(id);
                        if (!input || input.disabled) return;
                        input.checked = true;
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }});

                    let verified = 0;
                    selectedIds.forEach((id) => {{
                        const input = document.getElementById(id);
                        if (input && input.checked) verified += 1;
                    }});

                    return JSON.stringify({{ clicked, intended: selectedIds.length, verified, missing: Math.max(0, selectedIds.length - verified) }});
                }}"""
            )

            await asyncio.sleep(0.8)

            submit_result = await page.evaluate(
                """() => {
                    const hasUncheckedQuestion = Array.from(document.querySelectorAll('fieldset[id^="ef-question-"]')).some((fs) => {
                        return fs.querySelector('input[type="radio"]:checked, input[type="checkbox"]:checked') === null;
                    });
                    if (hasUncheckedQuestion) {
                        return JSON.stringify({ submitted: false, reason: 'unanswered-question' });
                    }

                    const submitCandidates = Array.from(document.querySelectorAll('input[type="submit"], button[type="submit"], button, a'));
                    const submitBtn = submitCandidates.find((el) => {
                        const txt = (el.textContent || el.value || '').trim().toLowerCase();
                        return txt === 'submit' || txt.includes('submit');
                    });

                    if (submitBtn && !submitBtn.disabled && submitBtn.getAttribute('aria-disabled') !== 'true') {
                        submitBtn.click();
                        return JSON.stringify({ submitted: true });
                    }
                    return JSON.stringify({ submitted: false });
                }"""
            )

            print(f"[Quiz Handler] Applied LLM-selected answers: {apply_result}; submit: {submit_result}")
            # Record this attempt so future retries can avoid repeating the same answers.
            if quiz_signature not in guard.quiz_attempts:
                guard.quiz_attempts[quiz_signature] = []
            guard.quiz_attempts[quiz_signature].append(list(selected_input_ids))
            guard.last_quiz_signature = quiz_signature
            guard.last_quiz_action_ts = time.time()
            await asyncio.sleep(2)

        except Exception as e:
            print(f"[Quiz Handler] Exception: {e}")
            pass  # Not a quiz or extraction failed

    async def guard_on_step_end(agent: Agent) -> None:
        page = await agent.browser_session.get_current_page()
        if page is None:
            return

        url = await agent.browser_session.get_current_page_url()
        if url == guard.last_url:
            guard.same_url_steps += 1
        else:
            guard.last_url = url
            guard.same_url_steps = 1
            guard.consecutive_skeleton_steps = 0
            guard.forced_reload_count = 0  # reset reload budget on each new page

        snapshot_raw = await page.evaluate(
            """() => {
                const visible = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
                };

                const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], input[type="submit"]'))
                    .filter(visible)
                    .map((el) => {
                        const value = (el.getAttribute('value') || '').trim();
                        const text = (el.textContent || value || '').trim().toLowerCase();
                        return {
                            text,
                            disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true'
                        };
                    });

                const hasSkeleton =
                    document.querySelector('[class*="skeleton"], [class*="loading"], [aria-busy="true"], .ph-item, .placeholder') !== null;

                const hasQuiz =
                    document.querySelector('fieldset[id^="ef-question-"], .que, form[id*="question"]') !== null;

                const bodyText = (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                const hasCourseCompletedText =
                    bodyText.includes('course completed') ||
                    bodyText.includes('congratulations') ||
                    bodyText.includes('you have completed this course');

                const hasCompletedOnlyButton = buttons.some(
                    (b) => !b.disabled && (b.text === 'completed' || b.text === 'course completed')
                );

                const hasComplete = buttons.some(
                    (b) => !b.disabled && (b.text.includes('complete and continue') || b.text === 'next' || b.text.includes('continue'))
                );
                const hasStartResume = buttons.some(
                    (b) => !b.disabled && (b.text.includes('resume course') || b.text === 'start' || b.text.includes("let's continue") || b.text.includes('continue course'))
                );
                return {
                    hasSkeleton,
                    hasQuiz,
                    hasCourseCompletedText,
                    hasCompletedOnlyButton,
                    hasComplete,
                    hasStartResume
                };
            }"""
        )

        try:
            snapshot = json.loads(snapshot_raw)
        except json.JSONDecodeError:
            return

        if snapshot.get("hasSkeleton"):
            guard.consecutive_skeleton_steps += 1
        else:
            guard.consecutive_skeleton_steps = 0
            guard.forced_reload_count = 0

        # Recover from persistent skeleton loading on the same page.
        # Never reload on the My Courses listing — it's slow by design and
        # reloading just restarts the scroll/expand cycle unnecessarily.
        is_on_my_courses_reload = "/start" in url or "/my-courses" in url or "/my_courses" in url
        if (
            not is_on_my_courses_reload
            and guard.consecutive_skeleton_steps >= 7
            and guard.same_url_steps >= 7
            and guard.forced_reload_count < 2
        ):
            print("Persistent loading skeleton detected; performing one hard reload.")
            await page.reload()
            await asyncio.sleep(6)
            guard.forced_reload_count += 1
            return

        # Deterministically prefer in-module progression buttons over navigation jumps.
        # Only fire when NOT on the My Courses landing page.
        is_on_my_courses = "/start" in url or "/my-courses" in url or "/my_courses" in url
        now = time.time()

        if (
            not is_on_my_courses
            and not snapshot.get("hasSkeleton")
            and not snapshot.get("hasQuiz")
            and (snapshot.get("hasCourseCompletedText") or snapshot.get("hasCompletedOnlyButton"))
            and (now - guard.last_completion_redirect_ts) > 3.0
        ):
            print("Course completion page detected; returning to home page.")
            await page.goto("https://learn.bostonifi.com/start")
            guard.last_completion_redirect_ts = now
            guard.last_url = "https://learn.bostonifi.com/start"
            guard.same_url_steps = 0
            guard.consecutive_skeleton_steps = 0
            guard.forced_reload_count = 0
            await asyncio.sleep(3)
            return

        if (
            not is_on_my_courses
            and not snapshot.get("hasSkeleton")
            and not snapshot.get("hasQuiz")
            and (snapshot.get("hasComplete") or snapshot.get("hasStartResume"))
            and (now - guard.last_forced_click_ts) > 2.5
        ):
            await page.evaluate(
                """() => {
                    const visible = (el) => {
                        if (!el) return false;
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
                    };

                    const controls = Array.from(document.querySelectorAll('button, a, input[type="button"], input[type="submit"]'))
                        .filter((el) => visible(el) && !el.disabled && el.getAttribute('aria-disabled') !== 'true');

                    const getText = (el) => ((el.textContent || el.getAttribute('value') || '').trim().toLowerCase());
                    const pick = (matcher) => controls.find((el) => matcher(getText(el)));

                    const target =
                        pick((t) => t.includes('complete and continue')) ||
                        pick((t) => t === 'next' || t === 'continue' || t.includes('continue')) ||
                        pick((t) => t.includes('resume course') || t === 'start' || t.includes("let's continue") || t.includes('continue course'));

                    if (target) {
                        target.click();
                        return 'clicked';
                    }
                    return 'no-target';
                }"""
            )
            guard.last_forced_click_ts = now
            await asyncio.sleep(2)

    async def combined_on_step_start(agent: Agent) -> None:
        """Run both disable_forbidden_nav and auto_handle_quiz at step start."""
        await enforce_allowed_domain(agent)
        await disable_forbidden_nav(agent)
        await auto_handle_quiz(agent)

    async def combined_on_step_end(agent: Agent) -> None:
        """Re-apply nav lock and then run guard at step end."""
        await enforce_allowed_domain(agent)
        await disable_forbidden_nav(agent)
        await guard_on_step_end(agent)

    def build_agent() -> Agent:
        return Agent(
            task=objective,
            llm=llm,
            fallback_llm=fallback_llm,
            tools=constrained_tools,
            include_attributes=["title", "aria-label", "id", "name", "role", "value", "checked", "aria-checked"],
            use_vision=True,
            directly_open_url=True,
            use_thinking=True,
            enable_planning=True,
            planning_replan_on_stall=2,
            loop_detection_window=12,
            max_actions_per_step=1,
            max_failures=20,
            final_response_after_failure=False,
            use_judge=False,
            available_file_paths=[],
            max_clickable_elements_length=120000,
            initial_actions=[],
        )

    last_error = None
    for attempt in range(1, 4):
        agent = build_agent()
        try:
            await agent.run(on_step_start=combined_on_step_start, on_step_end=combined_on_step_end)
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
