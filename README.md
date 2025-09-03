# Daily Journal Prompts

A tiny script that generates short, evocative journaling prompts as Markdown.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or: uv pip install -r requirements.txt
cp .env.example .env && $EDITOR .env  # add your OPENAI_API_KEY
python prompt_gen.py --count 5
```

## Example output:

```markdown

---
title: Daily Journal Prompts (2025-09-01)
model: gpt-5-nano
tone: gentle
style: minimalist
device: micro-story setup
seed: 20250901
---

# Daily Journal Prompts

- Reflection: – What did I notice today that I almost ignored?

- Learning: – Which small detail from today surprised my understanding?

- Joy: – Describe a moment when your hands found a quiet pleasure.

- Creativity: – Picture a scene where your sketchbook holds a new shape.

- Resilience: – Name a stubborn obstacle and the small step that pushed through.
```
