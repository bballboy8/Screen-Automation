import asyncio
import re
from browser_use import Agent
from browser_use.tools.service import Tools
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.messages import SystemMessage, UserMessage
from pydantic import BaseModel, Field
import json
import os


class QuestionAnswer(BaseModel):
    question_index: int = Field(description="1-based question index on the visible page")
    selected_option_indices: list[int] = Field(description="1-based option indices to select for this question")


class AnswerPlan(BaseModel):
    answers: list[QuestionAnswer]


async def extract_visible_question_form(page) -> dict | None:
    raw = await page.evaluate(
        r"""() => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };
            const buttonText = (el) => (el?.textContent || el?.value || '').replace(/\s+/g, ' ').trim().toLowerCase();
            const submitScore = (el) => {
                const text = buttonText(el);
                let score = 0;
                if (text.includes('submit')) score += 5;
                if (text.includes('continue')) score += 4;
                if (text.includes('complete')) score += 3;
                if (text.includes('next')) score += 2;
                if ((el?.id || '').toLowerCase().includes('submit')) score += 2;
                if ((el?.name || '').toLowerCase().includes('submit')) score += 2;
                return score;
            };
            const isValidActionButton = (el) => {
                const text = buttonText(el);
                if (!text) return false;
                if (
                    text.includes('start') ||
                    text.includes('repeat') ||
                    text.includes('retry') ||
                    text.includes('retake') ||
                    text.includes('resume')
                ) {
                    return false;
                }
                return text.includes('submit') || text.includes('continue') || text.includes('complete') || text.includes('next');
            };

            const root = document.querySelector('#main-content') || document;
            const fieldsets = Array.from(root.querySelectorAll("fieldset[id^='ef-question-']")).filter(isVisible);

            const submitCandidates = Array.from(document.querySelectorAll(
                "button[type='submit'], input[type='submit'], button[name='submit'], input[name='submit'], button[id*='submit' i], input[id*='submit' i], button[class*='submit' i], button.btn-primary, input.btn-primary, .btn.btn-primary, button.btn-success, input.btn-success"
            )).filter((el) => isVisible(el) && !el.disabled && isValidActionButton(el));
            const submitButton = submitCandidates.sort((a, b) => submitScore(b) - submitScore(a))[0] || null;
            if (!submitButton) {
                return JSON.stringify({ active: false });
            }

            const cleaned = (s) => (s || '').replace(/\s+/g, ' ').trim();
            const questionContainers = fieldsets.length ? fieldsets : [root];

            const questions = questionContainers.map((container, questionIndex) => {
                const inputs = Array.from(container.querySelectorAll("input[type='checkbox'], input[type='radio']"));
                const options = inputs.map((input, optionIndex) => {
                    const id = input.id || '';
                    // Prefer the explicit <label for="id"> text, then the wrapping label, then the next sibling
                    const explicitLabel = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
                    const wrapLabel = input.closest('label');
                    const sibling = input.nextElementSibling;
                    const visibleAnchor = explicitLabel || wrapLabel || sibling || input;
                    const text = cleaned(
                        explicitLabel?.textContent ||
                        wrapLabel?.textContent ||
                        sibling?.textContent ||
                        ''
                    );
                    return {
                        option_index: optionIndex + 1,
                        input_id: id,
                        input_type: input.type,
                        checked: !!input.checked,
                        visible: isVisible(visibleAnchor),
                        text,
                    };
                }).filter((option) => option.visible && option.text);

                if (!options.length) {
                    return null;
                }

                // Extract question text by cloning the fieldset and stripping inputs + labels,
                // leaving only the question stem text (works regardless of element structure).
                const clone = container.cloneNode(true);
                clone.querySelectorAll('input, label').forEach((el) => el.remove());
                const heading = cleaned(clone.textContent) || `Question ${questionIndex + 1}`;

                return {
                    question_index: questionIndex + 1,
                    question_text: heading,
                    selection_type: options.some((o) => o.input_type === 'checkbox') ? 'checkbox' : 'radio',
                    options,
                };
            }).filter((q) => q && q.options.length > 0);

            if (!questions.length) {
                return JSON.stringify({ active: false });
            }

            // Guard: require at least half the questions to have real question text (> 20 chars).
            // If not, the page hasn't fully loaded yet — skip to avoid a junk submission.
            const richCount = questions.filter((q) => q.question_text.length > 20).length;
            if (richCount < Math.ceil(questions.length / 2)) {
                return JSON.stringify({ active: false });
            }

            return JSON.stringify({
                active: true,
                submit_text: cleaned(submitButton?.textContent || submitButton?.value || ''),
                questions,
            });
        }"""
    )

    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if data.get("active") else None


async def build_answer_plan(agent: Agent, form_data: dict) -> AnswerPlan | None:
    questions = form_data.get("questions", [])
    if not questions:
        return None

    target_questions = [
        q for q in questions
        if not any(o.get("checked") for o in q.get("options", []))
    ] or questions

    if not target_questions:
        return None

    # Step 1: Ask the LLM to reason through each question as a CFP expert and
    # return final selected option indices. This supports multi-select checkboxes.
    reasoning_system = (
        "You are an expert Certified Financial Planner (CFP) with comprehensive mastery of: "
        "financial planning principles, estate planning (wills, trusts, powers of attorney, "
        "living wills, advance directives), insurance and risk management, investment planning, "
        "income tax planning, retirement planning, employee benefits, QDRO rules, divorce "
        "financial planning, Social Security, Medicare, Medicaid, monetary settlements, "
        "annuities, life insurance, disability insurance, deferred compensation, and CFP "
        "Code of Ethics and practice standards. "
        "You are answering questions from a CFP certification training course exam. "
        "Think carefully using your expert financial planning knowledge to identify the "
        "correct answer for each question. Be precise — these are professional exam questions."
    )

    q_lines = []
    for q in target_questions:
        selection_note = (
            "Select all correct options (multiple indices allowed)."
            if q.get("selection_type") == "checkbox"
            else "Select one best option (single index)."
        )
        opts = "\n".join(
            f"  Option {o['option_index']}: {o['text']}" for o in q["options"]
        )
        q_lines.append(
            f"Question {q['question_index']} ({q.get('selection_type', 'radio')}): {q['question_text']}\n"
            f"Instruction: {selection_note}\n{opts}"
        )
    q_block = "\n\n".join(q_lines)

    reasoning_user = (
        "Answer each financial planning exam question below correctly using your CFP expertise."
        f"\n\n{q_block}\n\n"
        "For each question respond in this exact format:\n"
        "Q<number>: <brief expert reasoning> → FINAL_INDICES: <comma-separated option numbers>\n"
        "Rules:\n"
        "- For radio questions, FINAL_INDICES must contain exactly one index.\n"
        "- For checkbox questions, include all correct indices and only correct indices.\n"
        "- Only use option numbers shown in the prompt."
    )

    try:
        raw = await agent.llm.ainvoke(
            [SystemMessage(content=reasoning_system), UserMessage(content=reasoning_user)]
        )
        if hasattr(raw, "content"):
            reasoning_text = str(raw.content)
        elif hasattr(raw, "completion"):
            reasoning_text = str(raw.completion)
        else:
            reasoning_text = str(raw)
    except Exception:
        return None

    if not reasoning_text:
        return None

    agent.logger.info(f"CFP expert reasoning:\n{reasoning_text}")

    # Step 2: Parse final selected option indices for each question.
    answers = []
    for q in target_questions:
        q_idx = q["question_index"]
        options = q["options"]
        is_checkbox = q.get("selection_type") == "checkbox"
        valid_indices = {opt["option_index"] for opt in options}

        # Isolate this question's section in the reasoning response
        section_match = re.search(
            rf"Q\s*{q_idx}\s*[:\.](.*?)(?=Q\s*\d+\s*[:\.]|$)",
            reasoning_text,
            re.DOTALL | re.IGNORECASE,
        )
        q_section = section_match.group(1) if section_match else reasoning_text

        indices_match = re.search(r"FINAL_INDICES:\s*([^\n]+)", q_section, re.IGNORECASE)
        indices_blob = indices_match.group(1) if indices_match else ""
        parsed_indices = [int(n) for n in re.findall(r"\d+", indices_blob)]
        selected_indices = [idx for idx in parsed_indices if idx in valid_indices]

        # Deduplicate while preserving order.
        selected_indices = list(dict.fromkeys(selected_indices))

        if is_checkbox:
            if not selected_indices and options:
                selected_indices = [options[0]["option_index"]]  # last-resort fallback
        else:
            selected_indices = selected_indices[:1]
            if not selected_indices and options:
                selected_indices = [options[0]["option_index"]]  # last-resort fallback

        answers.append(QuestionAnswer(question_index=q_idx, selected_option_indices=selected_indices))

    return AnswerPlan(answers=answers) if answers else None


async def apply_answer_plan(page, answer_plan: AnswerPlan) -> bool:
    raw = await page.evaluate(
        r"""(plan) => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };
            const buttonText = (el) => (el?.textContent || el?.value || '').replace(/\s+/g, ' ').trim().toLowerCase();
            const submitScore = (el) => {
                const text = buttonText(el);
                let score = 0;
                if (text.includes('submit')) score += 5;
                if (text.includes('continue')) score += 4;
                if (text.includes('complete')) score += 3;
                if (text.includes('next')) score += 2;
                if ((el?.id || '').toLowerCase().includes('submit')) score += 2;
                if ((el?.name || '').toLowerCase().includes('submit')) score += 2;
                return score;
            };
            const isValidActionButton = (el) => {
                const text = buttonText(el);
                if (!text) return false;
                if (
                    text.includes('start') ||
                    text.includes('repeat') ||
                    text.includes('retry') ||
                    text.includes('retake') ||
                    text.includes('resume')
                ) {
                    return false;
                }
                return text.includes('submit') || text.includes('continue') || text.includes('complete') || text.includes('next');
            };

            const root = document.querySelector('#main-content') || document;
            const fieldsets = Array.from(root.querySelectorAll("fieldset[id^='ef-question-']")).filter(isVisible);
            const submitCandidates = Array.from(document.querySelectorAll(
                "button[type='submit'], input[type='submit'], button[name='submit'], input[name='submit'], button[id*='submit' i], input[id*='submit' i], button[class*='submit' i], button.btn-primary, input.btn-primary, .btn.btn-primary, button.btn-success, input.btn-success"
            )).filter((el) => isVisible(el) && !el.disabled && isValidActionButton(el));
            const submitButton = submitCandidates.sort((a, b) => submitScore(b) - submitScore(a))[0] || null;
            if (!submitButton) {
                return JSON.stringify({ acted: false, reason: 'no_form' });
            }

            const clickInput = (input) => {
                try {
                    input.click();
                } catch {
                    input.checked = true;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            };

            const clickOption = (container, input) => {
                const id = input.id || '';
                const label = id
                    ? container.querySelector(`label[for="${CSS.escape(id)}"]`) || document.querySelector(`label[for="${CSS.escape(id)}"]`)
                    : null;
                if (label && isVisible(label)) {
                    try {
                        label.scrollIntoView({ block: 'center', inline: 'nearest' });
                        label.click();
                        return;
                    } catch {
                        // fall back to clicking the input directly
                    }
                }
                clickInput(input);
            };

            const getVisibleQuestionInputs = (container) => {
                return Array.from(container.querySelectorAll("input[type='checkbox'], input[type='radio']")).filter((input) => {
                    const id = input.id || '';
                    const explicitLabel = id ? container.querySelector(`label[for="${CSS.escape(id)}"]`) || document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
                    const wrapLabel = input.closest('label');
                    const sibling = input.nextElementSibling;
                    const visibleAnchor = explicitLabel || wrapLabel || sibling || input;
                    return isVisible(visibleAnchor);
                });
            };

            const getInputsForQuestion = (questionIndex) => {
                if (fieldsets.length) {
                    const fieldset = fieldsets[questionIndex - 1];
                    if (!fieldset) return { container: null, inputs: [] };
                    return {
                        container: fieldset,
                        inputs: getVisibleQuestionInputs(fieldset),
                    };
                }

                if (questionIndex !== 1) {
                    return { container: null, inputs: [] };
                }

                return {
                    container: root,
                    inputs: getVisibleQuestionInputs(root),
                };
            };

            let selectionsApplied = 0;
            let answeredQuestions = 0;

            for (const answer of plan.answers || []) {
                const { container, inputs } = getInputsForQuestion(answer.question_index);
                if (!container || !inputs.length) continue;
                const isCheckbox = inputs.some((input) => input.type === 'checkbox');

                if (isCheckbox) {
                    for (const input of inputs) {
                        if (input.checked) {
                            input.click();
                        }
                    }
                }

                for (const optionIndex of answer.selected_option_indices || []) {
                    const input = inputs[optionIndex - 1];
                    if (!input) continue;
                    if (!input.checked) {
                        clickOption(container, input);
                        if (input.checked) {
                            selectionsApplied += 1;
                        }
                    }
                }

                if (inputs.some((input) => input.checked)) {
                    answeredQuestions += 1;
                }
            }

            if (selectionsApplied === 0) {
                return JSON.stringify({ acted: false, reason: 'no_new_selections' });
            }

            if (answeredQuestions === 0) {
                return JSON.stringify({ acted: false, reason: 'no_answer_confirmed' });
            }

            try {
                submitButton.scrollIntoView({ block: 'center', inline: 'nearest' });
                submitButton.focus();
                submitButton.click();
                submitButton.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
            } catch {
                return JSON.stringify({ acted: false, reason: 'submit_failed' });
            }

            return JSON.stringify({ acted: true });
        }""",
        answer_plan.model_dump(),
    )

    try:
        result = json.loads(raw)
    except Exception:
        return False
    return bool(result.get("acted"))


async def try_matching_question(agent: Agent, page) -> bool:
    """Handle drag-and-drop matching questions using LLM reasoning."""
    raw = await page.evaluate(
        r"""() => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };

            // Look for matching question containers
            const leftItems = Array.from(document.querySelectorAll('[class*="left-cell"], [class*="left-item"], [class*="tl-left"]'))
                .filter(el => isVisible(el) && el.textContent.trim().length > 0);
            const rightItems = Array.from(document.querySelectorAll('[class*="right-cell"], [class*="right-item"], [class*="tl-right"]'))
                .filter(el => isVisible(el) && el.textContent.trim().length > 0);

            if (leftItems.length < 2 || rightItems.length < 2) {
                return JSON.stringify({ isMatching: false });
            }

            const questionText = document.querySelector('[class*="question"]')?.textContent || '';
            const isDragDrop = questionText.toLowerCase().includes('drag') || questionText.toLowerCase().includes('match');

            if (!isDragDrop && (leftItems.length === 0 || rightItems.length === 0)) {
                return JSON.stringify({ isMatching: false });
            }

            const leftData = leftItems.map((el, idx) => ({
                index: idx,
                text: el.textContent.trim(),
                id: el.id || `left-${idx}`,
                class: el.className
            }));

            const rightData = rightItems.map((el, idx) => ({
                index: idx,
                text: el.textContent.trim(),
                id: el.id || `right-${idx}`,
                class: el.className
            }));

            return JSON.stringify({
                isMatching: true,
                leftItems: leftData,
                rightItems: rightData,
                questionText: questionText.substring(0, 200)
            });
        }"""
    )

    try:
        match_info = json.loads(raw)
    except Exception:
        return False

    if not match_info.get("isMatching"):
        return False

    left_items = match_info.get("leftItems", [])
    right_items = match_info.get("rightItems", [])

    if not left_items or not right_items:
        return False

    # Ask LLM to reason about matches
    left_text = "\n".join([f"{i['text']}" for i in left_items])
    right_text = "\n".join([f"{i['text']}" for i in right_items])
    
    reasoning_prompt = (
        f"You are helping complete a matching question. "
        f"Match each LEFT item to the correct RIGHT item based on their meanings.\n\n"
        f"LEFT items:\n{left_text}\n\n"
        f"RIGHT items:\n{right_text}\n\n"
        f"For each left item, provide the matching right item text in this format:\n"
        f"LEFT_ITEM → RIGHT_ITEM_TEXT"
    )

    try:
        result = await agent.llm.ainvoke([UserMessage(content=reasoning_prompt)])
        if hasattr(result, "content"):
            reasoning = str(result.content)
        else:
            reasoning = str(result)
    except Exception:
        return False

    agent.logger.info(f"Matching question reasoning:\n{reasoning}")

    # Parse matches from LLM response
    matches = {}
    for line in reasoning.split('\n'):
        if '→' in line:
            parts = line.split('→')
            if len(parts) == 2:
                left_text_clean = parts[0].strip()
                right_text_clean = parts[1].strip()
                
                # Find matching indices
                left_idx = None
                right_idx = None
                
                for item in left_items:
                    if left_text_clean.lower() in item['text'].lower() or item['text'].lower() in left_text_clean.lower():
                        left_idx = item['index']
                        break
                
                for item in right_items:
                    if right_text_clean.lower() in item['text'].lower() or item['text'].lower() in right_text_clean.lower():
                        right_idx = item['index']
                        break
                
                if left_idx is not None and right_idx is not None:
                    matches[left_idx] = right_idx

    if not matches:
        return False

    # Perform drag operations for all matches using pointer events
    performed_any = await page.evaluate(
        r"""(matchPairs) => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };

            let performed = 0;
            const leftItems = Array.from(document.querySelectorAll('[class*="left-cell"], [class*="left-item"], [class*="tl-left"]'))
                .filter(el => isVisible(el));
            const rightItems = Array.from(document.querySelectorAll('[class*="right-cell"], [class*="right-item"], [class*="tl-right"]'))
                .filter(el => isVisible(el));

            for (const [leftIdx, rightIdx] of Object.entries(matchPairs)) {
                const leftEl = leftItems[parseInt(leftIdx)];
                const rightEl = rightItems[parseInt(rightIdx)];
                
                if (!leftEl || !rightEl) continue;

                try {
                    const leftRect = leftEl.getBoundingClientRect();
                    const rightRect = rightEl.getBoundingClientRect();

                    const leftX = leftRect.left + leftRect.width / 2;
                    const leftY = leftRect.top + leftRect.height / 2;
                    const rightX = rightRect.left + rightRect.width / 2;
                    const rightY = rightRect.top + rightRect.height / 2;

                    // Use pointer events which are more compatible with modern drag-drop
                    leftEl.dispatchEvent(new PointerEvent('pointerdown', {
                        bubbles: true,
                        cancelable: true,
                        pointerId: 1,
                        pointerType: 'mouse',
                        clientX: leftX,
                        clientY: leftY,
                        isPrimary: true
                    }));

                    leftEl.dispatchEvent(new MouseEvent('mousedown', {
                        bubbles: true,
                        cancelable: true,
                        clientX: leftX,
                        clientY: leftY
                    }));

                    // Simulate drag movement
                    leftEl.dispatchEvent(new PointerEvent('pointermove', {
                        bubbles: true,
                        cancelable: true,
                        pointerId: 1,
                        pointerType: 'mouse',
                        clientX: rightX,
                        clientY: rightY,
                        isPrimary: true
                    }));

                    rightEl.dispatchEvent(new PointerEvent('pointermove', {
                        bubbles: true,
                        cancelable: true,
                        pointerId: 1,
                        pointerType: 'mouse',
                        clientX: rightX,
                        clientY: rightY,
                        isPrimary: true
                    }));

                    rightEl.dispatchEvent(new PointerEvent('pointerover', {
                        bubbles: true,
                        cancelable: true,
                        pointerId: 1,
                        pointerType: 'mouse',
                        clientX: rightX,
                        clientY: rightY,
                        isPrimary: true
                    }));

                    rightEl.dispatchEvent(new DragEvent('dragover', {
                        bubbles: true,
                        cancelable: true,
                        clientX: rightX,
                        clientY: rightY
                    }));

                    rightEl.dispatchEvent(new DragEvent('drop', {
                        bubbles: true,
                        cancelable: true,
                        clientX: rightX,
                        clientY: rightY
                    }));

                    rightEl.dispatchEvent(new PointerEvent('pointerup', {
                        bubbles: true,
                        cancelable: true,
                        pointerId: 1,
                        pointerType: 'mouse',
                        clientX: rightX,
                        clientY: rightY,
                        isPrimary: true
                    }));

                    leftEl.dispatchEvent(new MouseEvent('mouseup', {
                        bubbles: true,
                        cancelable: true,
                        clientX: rightX,
                        clientY: rightY
                    }));

                    // Try to trigger change events if the widget uses those
                    rightEl.dispatchEvent(new Event('change', { bubbles: true }));
                    leftEl.dispatchEvent(new Event('change', { bubbles: true }));

                    performed++;
                } catch (e) {
                    console.error('Drag failed:', e);
                }
            }
            return performed;
        }""",
        matches
    )

    if performed_any > 0:
        agent.logger.info(f"Completed matching question via LLM-guided drag-and-drop ({performed_any} matches)")
        # Small delay to let the page update
        await asyncio.sleep(1)
        return True

    return False


async def try_start_course(page) -> bool:
    raw = await page.evaluate(
        r"""() => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };

            const candidates = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"], a'));
            const startButton = candidates.find((el) => {
                const text = (el.textContent || el.value || '').replace(/\s+/g, ' ').trim().toLowerCase();
                return isVisible(el) && text === 'start course';
            });

            if (!startButton) {
                return JSON.stringify({ acted: false, reason: 'no_start_button' });
            }

            if (startButton.disabled || startButton.getAttribute('aria-disabled') === 'true') {
                return JSON.stringify({ acted: false, reason: 'start_disabled' });
            }

            startButton.scrollIntoView({ block: 'center', inline: 'nearest' });
            startButton.click();
            return JSON.stringify({ acted: true });
        }"""
    )

    try:
        result = json.loads(raw)
    except Exception:
        return False
    return bool(result.get("acted"))


async def try_complete_survey(page) -> bool:
    raw = await page.evaluate(
        r"""() => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };
            const cleaned = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();

            const pageText = cleaned(document.body?.textContent || '');
            const looksLikeSurvey =
                pageText.includes('survey') ||
                pageText.includes('course evaluation') ||
                pageText.includes('feedback') ||
                pageText.includes('rate your') ||
                pageText.includes('evaluation');

            if (!looksLikeSurvey) {
                return JSON.stringify({ acted: false, reason: 'not_survey' });
            }

            let changed = 0;

            const clickSurveyOption = (input) => {
                const id = input.id || '';
                const label = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
                const visibleAnchor = label || input.closest('label') || input.nextElementSibling || input;
                if (!isVisible(visibleAnchor)) return false;
                visibleAnchor.click();
                return !!input.checked;
            };

            // Matrix/table surveys need one answer per visible row.
            const rowContainers = Array.from(document.querySelectorAll('tr, [role="row"], .survey-row, .matrix-row'))
                .filter((row) => isVisible(row));
            for (const row of rowContainers) {
                const radios = Array.from(row.querySelectorAll("input[type='radio']"));
                if (radios.length < 2) continue;
                if (radios.some((radio) => radio.checked)) continue;

                const visibleRadios = radios.filter((radio) => {
                    const id = radio.id || '';
                    const label = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
                    const visibleAnchor = label || radio.closest('label') || radio.nextElementSibling || radio;
                    return isVisible(visibleAnchor);
                });
                if (!visibleRadios.length) continue;

                const middle = visibleRadios[Math.floor((visibleRadios.length - 1) / 2)] || visibleRadios[0];
                if (middle && clickSurveyOption(middle)) {
                    changed += 1;
                }
            }

            const radioCheckboxes = Array.from(document.querySelectorAll("input[type='radio'], input[type='checkbox']"));
            const groupMap = new Map();

            for (const input of radioCheckboxes) {
                const id = input.id || '';
                const label = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
                const visibleAnchor = label || input.closest('label') || input.nextElementSibling || input;
                if (!isVisible(visibleAnchor)) continue;

                const groupKey = input.type === 'radio' ? `radio:${input.name || id}` : `checkbox:${id || Math.random()}`;
                if (!groupMap.has(groupKey)) {
                    groupMap.set(groupKey, []);
                }
                groupMap.get(groupKey).push({ input, visibleAnchor });
            }

            for (const [groupKey, options] of groupMap.entries()) {
                if (groupKey.startsWith('radio:')) {
                    if (options.some((option) => option.input.checked)) continue;
                    const middle = options[Math.floor((options.length - 1) / 2)] || options[0];
                    if (middle && !middle.input.checked) {
                        middle.visibleAnchor.click();
                        if (middle.input.checked) changed += 1;
                    }
                } else {
                    const first = options[0];
                    if (first && !first.input.checked) {
                        first.visibleAnchor.click();
                        if (first.input.checked) changed += 1;
                    }
                }
            }

            const selects = Array.from(document.querySelectorAll('select')).filter(isVisible);
            for (const select of selects) {
                if (select.value) continue;
                const validOptions = Array.from(select.options).filter((option) => option.value && cleaned(option.textContent) !== 'select one');
                if (!validOptions.length) continue;
                const pick = validOptions[Math.floor((validOptions.length - 1) / 2)] || validOptions[0];
                select.value = pick.value;
                select.dispatchEvent(new Event('input', { bubbles: true }));
                select.dispatchEvent(new Event('change', { bubbles: true }));
                changed += 1;
            }

            const textareas = Array.from(document.querySelectorAll('textarea')).filter(isVisible);
            for (const textarea of textareas) {
                if ((textarea.value || '').trim()) continue;
                textarea.value = 'N/A';
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                textarea.dispatchEvent(new Event('change', { bubbles: true }));
                changed += 1;
            }

            const candidates = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"], a'))
                .filter((el) => isVisible(el) && !el.disabled && el.getAttribute('aria-disabled') !== 'true');
            const submit = candidates.find((el) => {
                const text = cleaned(el.textContent || el.value || '');
                return text.includes('submit') || text.includes('continue') || text.includes('next') || text.includes('complete');
            });

            if (!submit) {
                return JSON.stringify({ acted: changed > 0, reason: 'survey_filled_no_submit' });
            }

            if (changed === 0) {
                return JSON.stringify({ acted: false, reason: 'no_survey_changes' });
            }

            submit.scrollIntoView({ block: 'center', inline: 'nearest' });
            submit.click();
            return JSON.stringify({ acted: true, reason: 'survey_completed' });
        }"""
    )

    try:
        result = json.loads(raw)
    except Exception:
        return False
    return bool(result.get("acted"))


async def try_progress_button(page) -> bool:
    raw = await page.evaluate(
        r"""() => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };
            const cleaned = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();

            const hasQuestionInputs = Array.from(document.querySelectorAll("input[type='checkbox'], input[type='radio']")).length > 0;
            const hasQuestionSubmit = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"]'))
                .filter(isVisible)
                .some((el) => {
                    const text = cleaned(el.textContent || el.value || '');
                    const id = cleaned(el.id || '');
                    const name = cleaned(el.getAttribute('name') || '');
                    return text === 'submit' || id.includes('submit-question') || name === 'submit';
                });
            if (hasQuestionInputs || hasQuestionSubmit) {
                return JSON.stringify({ acted: false, reason: 'question_page' });
            }

            const candidates = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"], a'))
                .filter((el) => isVisible(el) && !el.disabled && el.getAttribute('aria-disabled') !== 'true');

            const scoreButton = (el) => {
                const text = cleaned(el.textContent || el.value || '');
                let score = -1;
                if (text.includes('complete and continue')) score = 10;
                else if (text.includes("completed. let's continue")) score = 9;
                else if (text === 'complete') score = 8;
                else if (text.includes('continue')) score = 7;
                else if (text.includes('next unit')) score = 6;
                else if (text.includes('next')) score = 5;
                return score;
            };

            const progressButton = candidates
                .map((el) => ({ el, score: scoreButton(el) }))
                .filter((item) => item.score >= 0)
                .sort((a, b) => b.score - a.score)[0];

            if (!progressButton) {
                return JSON.stringify({ acted: false, reason: 'no_progress_button' });
            }

            progressButton.el.scrollIntoView({ block: 'center', inline: 'nearest' });
            progressButton.el.click();
            return JSON.stringify({ acted: true });
        }"""
    )

    try:
        result = json.loads(raw)
    except Exception:
        return False
    return bool(result.get("acted"))


async def try_open_incomplete_tree_item(page) -> bool:
    raw = await page.evaluate(
        r"""() => {
            const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
            };
            const cleaned = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();

            const mainContent = document.querySelector('#main-content');
            const mainText = cleaned(mainContent?.textContent || '');
            const hasQuestionForm = Array.from(document.querySelectorAll("input[type='checkbox'], input[type='radio']")).length > 0 ||
                Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"]'))
                    .filter(isVisible)
                    .some((el) => {
                        const text = cleaned(el.textContent || el.value || '');
                        const id = cleaned(el.id || '');
                        return text === 'submit' || id.includes('submit-question');
                    });
            const progressButtonsVisible = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"], a'))
                .filter(isVisible)
                .some((el) => {
                    const text = cleaned(el.textContent || el.value || '');
                    return text.includes('complete and continue') || text === 'complete' || text.includes("completed. let's continue") || text.includes('next unit');
                });
            const pageLooksCompleted =
                mainText.includes('current unit - passed') ||
                mainText.includes('questions answered') ||
                mainText.includes('passed with') ||
                mainText.includes('not passed') ||
                mainText === 'sources' ||
                mainText.startsWith('sources ');

            // If it's a question form, don't look for tree completion
            if (hasQuestionForm) {
                return JSON.stringify({ acted: false, reason: 'has_question_form' });
            }

            // Check if page shows test/unit completion (Passed, etc.)
            // Even if progress buttons are visible, if the page shows completion, offer tree navigation
            if (!pageLooksCompleted) {
                return JSON.stringify({ acted: false, reason: 'page_not_completed' });
            }

            const treeItems = Array.from(document.querySelectorAll('[role="treeitem"]')).filter(isVisible);
            if (!treeItems.length) {
                return JSON.stringify({ acted: false, reason: 'no_tree_items' });
            }

            const isSkippable = (text) => (
                text.includes('supplemental resource') ||
                text.includes('qbank') ||
                text.includes('optional') ||
                text.includes('practice')
            );

            const isCompleted = (text) => (
                text.includes('passed') ||
                text.includes('completed')
            );

            const candidates = treeItems
                .map((el) => {
                    const text = cleaned([
                        el.textContent,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                        el.getAttribute('class'),
                    ].filter(Boolean).join(' '));
                    return { el, text };
                })
                .filter((item) => item.text && !isSkippable(item.text) && !isCompleted(item.text));

            const preferred = candidates.find((item) => item.text.includes('current unit') || item.text.includes('current')) || candidates[0];
            if (!preferred) {
                return JSON.stringify({ acted: false, reason: 'no_incomplete_tree_item' });
            }

            preferred.el.scrollIntoView({ block: 'center', inline: 'nearest' });
            preferred.el.click();
            return JSON.stringify({ acted: true, target: preferred.text });
        }"""
    )

    try:
        result = json.loads(raw)
    except Exception:
        return False
    return bool(result.get("acted"))


def build_on_step_end_callback():
    async def on_step_end(agent: Agent) -> None:
        page = await agent.browser_session.get_current_page()
        if page is None:
            return

        # Wait for the page to settle after each agent action before inspecting the form.
        # This prevents firing on splash/start screens before questions have rendered.
        await asyncio.sleep(2)

        if await try_start_course(page):
            agent.logger.info("Auto-clicked enabled Start course button")
            return

        # Try to extract and answer question forms
        # This ensures quiz/exam questions get answered even if surveys are also present.
        form_data = await extract_visible_question_form(page)
        if form_data:
            answer_plan = await build_answer_plan(agent, form_data)
            if answer_plan and answer_plan.answers:
                applied = await apply_answer_plan(page, answer_plan)
                if applied:
                    agent.logger.info("LLM-guided helper selected visible answers and submitted the form")
                    return

        if await try_complete_survey(page):
            agent.logger.info("Auto-completed required survey/evaluation form")
            return

        if await try_progress_button(page):
            agent.logger.info("Auto-clicked visible progress button")
            return

        if await try_open_incomplete_tree_item(page):
            agent.logger.info("Opened first visible incomplete tree item after landing on a completed section")
            return

    return on_step_end

async def main():
    username = os.getenv("BOSTONIFI_USERNAME")
    password = os.getenv("BOSTONIFI_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Missing credentials. Set BOSTONIFI_USERNAME and BOSTONIFI_PASSWORD environment variables."
        )

    # 1. Initialize Gemini
    # Primary model: higher reliability for custom course widgets. Fallback model: lower cost.
    llm = ChatGoogle(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    fallback_llm = ChatGoogle(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )

    # 2. Define the objective
    # Be very specific about what you want it to do.
    objective = (
        "Open https://learn.bostonifi.com/start and log in with the provided credentials. "
        "If the first page is blank, navigate again to the same URL immediately. "
        f"Use this exact username: {username} and this exact password: {password}. "
        "Do not add any extra characters, punctuation, or spaces to username or password. "
        "If you ever land on a password-change page before login succeeds, navigate back to https://learn.bostonifi.com/start and try login again. "
        "On My Courses, complete all required incomplete items across the full program. "
        "Process courses and modules strictly in displayed order (top to bottom, lowest to highest), and do not skip ahead. "
        "Never skip a required incomplete item just because a Start course button is temporarily disabled or the page is still loading. "
        "If a required exam or course shows Start course disabled, treat that as a temporary loading problem: wait, refresh or re-open the same required item, and keep retrying until it becomes available. "
        "Do not move on to a later required item while an earlier required item is still incomplete, even if its Start course button is disabled. "
        "On My Courses, never click placeholder/skeleton tiles or symbol-only entries such as '*', '...', or empty text. "
        "Before clicking a course/module card, verify the target element text contains the actual course or module name words. "
        "If a click target text is generic or symbol-only, do not click it; instead use find_elements with include_text=true and click the returned index whose visible text best matches the required course/module title. "
        "If the page appears to be loading skeleton content, wait and re-identify elements before clicking again. "
        "Within each course, complete units and assessments in order before moving to the next module. "
        "Skip completed items and skip Supplemental Resource, QBank, Optional, and Practice items. "
        "If a required survey, evaluation, or feedback form appears, complete it and continue; do not get stuck on surveys. "
        "After opening any individual course or module page, click 'Resume course' before attempting lesson or test actions. "
        "If clicking 'Resume course' lands on a completed unit such as Sources, Summary, or a Passed page, inspect the left course tree and open the first visible required unit or section that is not marked Passed or Completed. "
        "When inside a course tree, use the tree to locate unfinished items before deciding the module is complete. "
        "Use normal UI actions first (click, scroll, input, drag-and-drop). "
        "For answering questions, do not use JavaScript evaluate scripts unless there is absolutely no other option. "
        "When answering questions, prefer clicking the visible answer label, button, or option container that contains the answer text; do not rely on raw input value selectors unless visible clicks fail. "
        "On quiz or exam question pages, do not use manual CSS selector tricks such as label:contains, and do not manually click the Submit button if the helper is already selecting answers; let the question helper handle answer selection and submission. "
        "If an element index becomes stale or unavailable, refresh browser state and re-identify the visible answer option before submitting. "
        "For radio and checkbox questions, first use find_elements with include_text=true to locate the interactive options, then click the exact indices for the matching visible answers. "
        "If find_elements returns the options, click those returned interactive indices directly instead of clicking fieldsets or generic containers. "
        "Do not use unsupported CSS selectors such as :contains in find_elements. "
        "If label or option text is visible in find_elements results, click that returned label/option index directly instead of switching to JavaScript. "
        "Do not repeat find_elements on the same selector more than once in a row without clicking one of the returned indices. "
        "After identifying answer options, immediately click the needed returned indices in the same step or the very next step. "
        "If a question has multiple unanswered parts visible on screen, answer all visible parts before submitting. "
        "If a module test shows Passed, treat it as complete and move on; do not repeat passed tests just to chase 100%. "
        "For matching/drag screens, use real drag-and-drop and verify each item is placed before submitting. "
        "For drag-and-drop matching questions, identify all left-side items and right-side descriptions, then use drag action to drag each left item to its matching right item. "
        "Each item must show a visual confirmation (arrow, connecting line, or checkmark) before you submit the answer. Do not submit until all items are properly matched and confirmed. "
        "For 'select all that apply' questions, select only the options supported by the question text; never click every option. "
        "Before submitting any test, verify each question has an intentional answer selection and no accidental extra selections. "
        "Never click Submit on a question page unless at least one visible answer choice is selected. "
        "After each submit, click the next/continue button that appears. "
        "After answering a set of questions, stay in the current module and click the visible green progress button (Continue, Submit, or Complete and continue) to move forward. "
        "If a test fails, click Repeat test (or Retry/Retake) and try again until passed. "
        "After clicking Repeat test, wait for the refreshed question form to fully render, then select answers again using currently visible elements. "
        "Do NOT return to My Courses after question sets or test completion. STAY within the current course and use the left course tree to navigate to the next incomplete unit. "
        "After a test passes or question set completes, look for a progress button (Complete and continue, Next, Continue), click it, and it will advance you within the course. "
        "Use the course tree on the left side to identify and click the next incomplete required unit or section. "
        "Only return to My Courses when the entire current course is 100% complete with no incomplete items visible in the tree. "
        "When you land on a completed unit page (Sources, Summary, Passed), use try_open_incomplete_tree_item helper or manually click the first incomplete item in the tree to continue. "
        "Do not create or edit local files. Do not read or write any files or use file system actions. Keep all planning in working memory only. "
        "Do not call done while any required item remains incomplete. "
        "Only finish when the full program is complete and final completion proof is visible."
    )

    # 3. Initialize the Agent
    constrained_tools = Tools(exclude_actions=["evaluate", "write_file", "replace_file", "read_file"])

    agent = Agent(
        task=objective,
        llm=llm,
        fallback_llm=fallback_llm,
        tools=constrained_tools,
        include_attributes=["title", "aria-label", "id", "class", "name", "role", "href", "value"],
        use_vision=True, # Critical: This lets Gemini 'see' the buttons
        directly_open_url=True,
        use_thinking=False,
        enable_planning=False,
        max_actions_per_step=6,
        available_file_paths=[],  # Disable all file I/O for the agent
        initial_actions=[
            {"navigate": {"url": "https://learn.bostonifi.com/start", "new_tab": False}}
        ],
    )

    # 4. Run the agent
    await agent.run(on_step_end=build_on_step_end_callback())

if __name__ == "__main__":
    asyncio.run(main())