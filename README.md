# Screen Automation

Minimal browser automation scripts using `browser-use` + Gemini.

## What is in this repo

- `course_bot.py`: Current production workflow for the Boston iFi course flow.
- `process_bot_template.py`: Tiny starter script for new automations.
- `requirements.txt`: Pinned Python dependencies.

## Quick start

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Set environment variables.

```bash
export GOOGLE_API_KEY="your_google_api_key"
```

For `course_bot.py`, also set:

```bash
export BOSTONIFI_USERNAME="your_username"
export BOSTONIFI_PASSWORD="your_password"
```

4. Run a script.

```bash
python3 course_bot.py
```

Or start from the minimal template:

```bash
python3 process_bot_template.py
```

## Creating a new automation quickly

1. Copy `process_bot_template.py` to a new file.
2. Update the `objective` string with the new process goal.
3. Update the initial URL in `initial_actions`.
4. Run the script and iterate.

## Notes

- The scripts use visual browser interaction (`use_vision=True`).
- Keep objectives explicit and step-focused for best results.
- Never commit real credentials.
