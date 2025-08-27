# Daily Journal Prompts

A tiny script that generates short, evocative journaling prompts as Markdown.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or: uv pip install -r requirements.txt
cp .env.example .env && $EDITOR .env  # add your OPENAI_API_KEY
python prompt_gen.py --count 5
