# OpenAgent

Terminal AI coding assistant running on local LLMs via [LM Studio](https://lmstudio.ai/). Your code never leaves your machine.

You describe a task, the agent reads files, writes code, runs commands, fixes bugs on its own. If it doesn't know how — it searches the web via DuckDuckGo.

## Quick Start

You need Python 3.9+ and [LM Studio](https://lmstudio.ai/) running with a loaded model.

```bash
git clone https://github.com/amberoad666/OpenAgent.git
cd OpenAgent
pip install -r requirements.txt
python main.py
```

Or use the installer (adds to PATH):

```bash
# Windows
install.bat

# Linux / macOS
chmod +x install.sh && ./install.sh
```

After that you can run `open-agent` or `oa` from any directory.

## What It Does

You give it a task — it does the work. Doesn't ask for clarifications, doesn't give you instructions like "create file X" — it calls tools and creates files itself.

```
> build a Flask REST API for a todo app with SQLite

> find bugs in my project and fix them

> add dark mode to index.html using Tailwind
```

It works in a loop: calls a tool → gets the result → decides what to do next → repeats until done. Confirmation is only asked for file writes and shell commands (disable with `/yolo`).

## Models

Tested models:

- **qwen3.5-35b-a3b** — best results, gets it right on the first try
- **qwen2.5-coder-32b** — strong at writing code
- **deepseek-coder-v2** — also works well

Smaller models (7B, 14B) will run too but expect more mistakes and corrections.

## Commands

```
/yolo          — skip all confirmations
/add <file>    — pin file to context (survives compaction)
/add           — list pinned files
/undo          — revert last file change
/model         — switch model
/cost          — token usage and speed stats
/save-session  — save conversation
/load-session  — load conversation
/clear         — clear history
```

## Configuration

Environment variables (all optional):

```bash
OPEN_AGENT_URL=http://localhost:1234/v1   # LM Studio endpoint
OPEN_AGENT_MODEL=qwen/qwen3.5-35b-a3b    # force a specific model
OPEN_AGENT_MAX_TOKENS=8192               # context window size
```

A `.openagent` file in your project root provides persistent instructions the agent loads automatically:

```
This is a Django project with PostgreSQL.
Use pytest for tests.
```

## How It Works

```
main.py          — core loop, tool call parsing, streaming
config.py        — system prompt, config
client.py        — LM Studio API client
executor.py      — tool execution
tools.py         — tool implementations (files, bash, search)
ui.py            — terminal UI (Rich)
indexer.py       — TF-IDF codebase index for auto-context
input_support.py — tab completion, command highlighting
learning.py      — few-shot memory and DPO pair collection
```

LM Studio models can't reliably use the OpenAI `tools` parameter in streaming mode — they return empty arguments. So OpenAgent puts the call format in the system prompt, the model writes tool calls as plain text, and a parser extracts them from 6 different formats.

When the model misbehaves (gives instructions instead of acting, asks the user to share a file, says "I created the file" without actually calling a tool) — a detector catches it and sends a nudge with the correct format. Three escalation levels, then stop.

Clean runs (0 nudges, 0 errors) are saved as few-shot examples in `data/few_shot.jsonl`. Bad runs are saved as DPO pairs in `data/dpo_pairs.jsonl`.

## License

MIT
