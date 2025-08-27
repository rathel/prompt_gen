import os
import configparser
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

config = configparser.ConfigParser()
config.read("config.ini")

DATE = datetime.now().strftime("%Y-%m-%d")

DIRECTORY = Path(config.get("Settings", "directory", fallback="prompts")).expanduser()

FILE = DIRECTORY / f"{DATE}.md"

DIRECTORY.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=config.get("Settings", "api_key", fallback=os.getenv("OPENAI_API_KEY")))

response = client.responses.create(
  model="gpt-5-nano",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "You are a journaling coach. Generate concise, evocative prompts.\nVary themes (reflection, gratitude, goals, creativity, shadow work)."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Always use markdown dash-style lists instead of numbered lists.\nNow make 5 daily journal prompts.\nFor each prompt prepend theme with a colon. For example:\nReflection:\nGratitude:\nGoals:\nCreativity:\nShadow Work:"
        }
      ]
    },
  ],
  text={
    "format": {
      "type": "text"
    },
    "verbosity": "low"
  },
  reasoning={
    "effort": "low"
  },
  tools=[],
  store=True
)

text = getattr(response, "output_text", None)
text = text.splitlines() if text else []
text = "\n\n".join(text)

if not text:
    d = response.to_dict()
    chunks = []
    for item in d.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("text", "output_text"):
                    chunks.append(c.get("text", ""))
    text = "\n".join(chunks).strip()

with open(FILE, "w") as f:
    f.write("# Daily Journal Prompts\n\n")
    f.write(text or "No prompts generated.\n")

