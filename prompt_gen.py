#!/usr/bin/env python3
import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
import configparser
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# CLI & config
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate daily journaling prompts as Markdown.")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"), help="OpenAI model name")
    p.add_argument("--count", type=int, default=5, help="How many prompts to generate")
    p.add_argument("--dir", dest="directory", default=None, help="Output directory (overrides config)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: date-based)")
    p.add_argument("--cool", action="store_true", help="Cooler settings (less variety)")
    p.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    return p.parse_args()


def setup_logging(verbosity: int):
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# -----------------------------
# Variety pools
# -----------------------------
THEMES = [
    "Reflection", "Gratitude", "Goals", "Creativity", "Shadow Work",
    "Mindfulness", "Relationships", "Self‑Compassion", "Resilience", "Values",
    "Learning", "Career", "Health", "Joy", "Curiosity", "Boundaries",
]

TONES = ["gentle", "provocative", "playful", "stoic", "poetic", "coach‑y"]
STYLES = ["Socratic questions", "metaphor‑rich", "minimalist", "sensory‑focused", "future‑self", "counterfactual"]
DEVICES = ["if/then riff", "3‑minute sprint", "first‑line finish", "contrast two lists", "micro‑story setup"]


# -----------------------------
# Prompt builders
# -----------------------------

def build_messages(n: int, theme_choices, tone, style, device, cooler: bool):
    """Builds a developer+user message with explicit output format rules."""
    # Stable system guidance improves consistency across samples
    developer_text = (
        "You are a journaling coach. Generate concise, evocative prompts.\n"
        "Vary themes (reflection, gratitude, goals, creativity, shadow work, and more).\n"
        "Prefer concrete imagery over abstractions; avoid therapy jargon."
    )

    # Output contract
    format_rules = (
        "Always use markdown dash-style lists (\"- \"), never numbers.\n"
        "Each item must begin with a Theme label followed by a colon, then the prompt.\n"
        "Example line: 'Reflection: – What did I notice today that I almost ignored?'\n"
        "Keep each prompt <= 20 words. No filler, no preambles."
    )

    # Style knobs
    style_knobs = (
        f"Tone: {tone}. Style: {style}. Device: {device}.\n"
        "Favor specific nouns/verbs; skip cliches like 'journey' or 'authentic'."
    )

    # Anti-repetition nudges
    dedupe_rules = (
        "Avoid repeating verbs, openings, or structures within the set.\n"
        "Vary sentence openings across the list."
    )

    # Temperature profile
    variety_hint = "Keep it grounded and vivid." if cooler else "Take tasteful risks and surprise me."

    # Compose user instruction
    user_text = (
        f"Make {n} daily journal prompts.\n"
        f"Chosen themes (shuffle and use each once): {', '.join(theme_choices)}.\n"
        f"{format_rules}\n{style_knobs}\n{dedupe_rules}\n{variety_hint}"
    )

    messages = [
        {"role": "developer", "content": [{"type": "input_text", "text": developer_text}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
    ]
    return messages


# -----------------------------
# Output helpers
# -----------------------------

def extract_text(response) -> str:
    # Prefer the SDK convenience if present
    text = getattr(response, "output_text", None)
    if text:
        # Keep markdown spacing consistent
        lines = text.splitlines()
        return "\n\n".join(lines)

    # Fallback — Responses API shape
    try:
        d = response.to_dict()
    except Exception:
        return ""

    chunks = []
    for item in d.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("text", "output_text"):
                    chunks.append(c.get("text", ""))
    return "\n".join(chunks).strip()


# -----------------------------
# History (light anti-duplication)
# -----------------------------

def load_recent(history_path: Path, limit: int = 1000):
    items = []
    if history_path.exists():
        with history_path.open() as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    # Return the most recent text bodies for quick substring checks
    return [it.get("text", "") for it in items][-limit:]


def looks_too_similar(new_text: str, recents: list[str]) -> bool:
    # Super light check: if any recent prompt chunk shares a long 6-word n-gram, re-roll
    new_grams = set(" ".join(new_text.split()[i:i+6]).lower() for i in range(max(0, len(new_text.split()) - 5)))
    for blob in recents:
        grams = set(" ".join(blob.split()[i:i+6]).lower() for i in range(max(0, len(blob.split()) - 5)))
        if new_grams & grams:
            return True
    return False


def append_history(history_path: Path, payload: dict):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    setup_logging(args.verbose)

    load_dotenv()  # allow env overrides

    # Read config.ini (optional)
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")

    directory = (
        Path(args.directory).expanduser() if args.directory
        else Path(cfg.get("Settings", "directory", fallback="prompts")).expanduser()
    )
    directory.mkdir(parents=True, exist_ok=True)

    api_key = cfg.get("Settings", "api_key", fallback=os.getenv("OPENAI_API_KEY"))
    if not api_key:
        print("OPENAI_API_KEY not configured (config.ini or environment)", file=sys.stderr)
        sys.exit(2)

    client = OpenAI(api_key=api_key)

    # Seed: default to YYYYMMDD for daily consistency, override with --seed for reproducibility
    seed = args.seed if args.seed is not None else int(datetime.now().strftime("%Y%m%d"))
    random.seed(seed)

    # Choose themes & style knobs
    theme_choices = random.sample(THEMES, k=args.count)
    random.shuffle(theme_choices)
    tone = random.choice(TONES)
    style = random.choice(STYLES)
    device = random.choice(DEVICES)

    # Sampling profile
    if args.cool:
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "max_output_tokens": 400,
        }
    else:
        generation_config = {
            "temperature": 1.15,
            "top_p": 0.96,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.35,
            "max_output_tokens": 400,
        }

    # Build messages
    messages = build_messages(args.count, theme_choices, tone, style, device, args.cool)

    # Call the API
    logging.info("Requesting prompts... (tone=%s, style=%s, device=%s)", tone, style, device)
    response = client.responses.create(
        model=args.model,
        input=messages,
        text={"format": {"type": "text"}},
        reasoning={"effort": "low"},
        generation_config=generation_config,
        store=True,
    )

    text = extract_text(response)

    # Very light duplicate guard: if too similar to recent history, try one quick re-roll
    history_path = directory / ".history.jsonl"
    recents = load_recent(history_path)
    if looks_too_similar(text, recents):
        logging.info("Output resembled recent history — re‑rolling once for variety…")
        response = client.responses.create(
            model=args.model,
            input=messages,
            text={"format": {"type": "text"}},
            reasoning={"effort": "low"},
            generation_config=generation_config,
            store=True,
        )
        text = extract_text(response)

    # Write Markdown (with light front matter for GitHub renderers)
    date_str = datetime.now().strftime("%Y-%m-%d")
    outfile = directory / f"{date_str}.md"
    with outfile.open("w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"title: Daily Journal Prompts ({date_str})\n")
        f.write(f"model: {args.model}\n")
        f.write(f"tone: {tone}\nstyle: {style}\ndevice: {device}\n")
        f.write(f"seed: {seed}\n")
        f.write("---\n\n")
        f.write("# Daily Journal Prompts\n\n")
        f.write(text or "No prompts generated.\n")

    # Save raw to history for future de‑duplication
    append_history(history_path, {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "tone": tone,
        "style": style,
        "device": device,
        "seed": seed,
        "text": text,
    })

    print(outfile)


if __name__ == "__main__":
    main()
