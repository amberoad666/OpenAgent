# OpenAgent

Terminal AI coding assistant powered by local LLMs via [LM Studio](https://lmstudio.ai/).

OpenAgent reads, writes, and edits files, runs shell commands, searches the web, and manages your codebase — all from the terminal, with no cloud API needed.

## Features

- **Fully local** — runs against LM Studio on `localhost`, your code never leaves your machine
- **Autonomous agent** — doesn't ask questions, uses tools to find information itself
- **7 built-in tools** — `write_file`, `read_file`, `edit_file`, `run_bash`, `list_files`, `search_files`, `web_search`
- **Universal tool parser** — handles multiple LLM output formats (`<function=...>`, `<tool_call>`, Python-style calls, etc.)
- **Streaming output** — real-time response with syntax-highlighted code blocks
- **Smart behavior detection** — catches and corrects 5 types of bad model behavior (tutorial mode, lazy responses, fake actions, shell-in-text, duplicates)
- **Codebase RAG** — auto-indexes your project and injects relevant files as context
- **Self-learning** — collects few-shot examples from successful runs and DPO pairs from corrections
- **Session management** — save/load conversation sessions
- **Persistent /add** — pin files to context that survive conversation compaction
- **File undo** — revert the last file change with `/undo`
- **YOLO mode** — skip all confirmation prompts for faster iteration

## Requirements

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running with a loaded model
- Recommended models:
  - `qwen/qwen3.5-35b-a3b` — best results, passes quality checks on first attempt
  - `qwen2.5-coder-32b` — strong coding model
  - `deepseek-coder-v2` — good alternative

## Installation

### Windows

```bash
git clone https://github.com/user/OpenAgent.git
cd OpenAgent
install.bat
```

This adds OpenAgent to your PATH and installs dependencies. After restarting the terminal:

```bash
open-agent
# or
oa
```

### Linux / macOS

```bash
git clone https://github.com/user/OpenAgent.git
cd OpenAgent
chmod +x install.sh oa open-agent
./install.sh
```

Then:

```bash
open-agent
# or
oa
```

### Manual

```bash
pip install rich prompt_toolkit
python main.py
```

## Usage

1. Start LM Studio and load a model
2. Run `open-agent` (or `oa`, or `python main.py`)
3. Type your task in natural language

```
> create a Flask REST API for a todo app with SQLite

> find and fix bugs in my project

> add dark mode to index.html using Tailwind
```

### Keyboard

| Key | Action |
|---|---|
| `Enter` | Send message |
| `\` + `Enter` | New line |
| `Esc` + `Enter` | Send multiline message |
| `Right Arrow` | Accept suggested next action |
| `Esc` | Stop current generation |
| `Ctrl+C` | Stop running command |

### Commands

| Command | Description |
|---|---|
| `/help` | Show help |
| `/yolo` | Toggle skip-permissions mode |
| `/add <path>` | Pin file/dir/glob to context (persists through compaction) |
| `/add` | List pinned files |
| `/undo` | Undo last file change |
| `/save` | Save code blocks from last response |
| `/model` | Switch to a different loaded model |
| `/cost` | Show session token usage and speed stats |
| `/learn` | Show self-learning statistics |
| `/reindex` | Rebuild codebase index |
| `/save-session [name]` | Save conversation to file |
| `/load-session [name]` | Load a saved conversation |
| `/clear` | Clear conversation history |
| `/exit` | Exit |

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `OPEN_AGENT_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `OPEN_AGENT_API_KEY` | `lm-studio` | API key |
| `OPEN_AGENT_MODEL` | _(auto-detect)_ | Force a specific model |
| `OPEN_AGENT_MAX_TOKENS` | `8192` | Max context window size |
| `OPEN_AGENT_DEBUG` | _(off)_ | Set to `1` to enable debug logging to `debug.log` |

### Project Memory

Create a `.openagent` file in your project root to give the agent persistent instructions:

```
This is a Django project using PostgreSQL.
Always use pytest for tests.
Follow PEP 8 style.
```

OpenAgent will automatically load this file on startup and after `/clear`.

## Architecture

```
main.py          Core agent loop, tool parsing, streaming, behavior detection
config.py        System prompt, API config, rules
client.py        LM Studio API client (chat, streaming, summarization)
executor.py      Tool execution with confirmation flow
tools.py         Tool implementations (file ops, bash, web search)
ui.py            Rich terminal UI (panels, diffs, syntax highlighting)
indexer.py       TF-IDF codebase indexer for auto-context
input_support.py Tab completion, command highlighting, auto-suggest
learning.py      Few-shot memory + DPO data collection
```

### How Tool Calling Works

LM Studio models don't reliably support the OpenAI `tools` API parameter in streaming mode. Instead, OpenAgent:

1. Shows the model tool definitions in the system prompt with example format
2. The model emits tool calls as **text** in its response
3. `_parse_tool_calls_from_text()` extracts tool calls from 6 different formats
4. Stream output is suppressed once a tool call pattern is detected
5. Tools execute with user confirmation (or auto in YOLO mode)
6. Results are sent back as user messages for the next turn

### Behavior Detection

OpenAgent detects and corrects common LLM failure modes:

| Detector | What it catches |
|---|---|
| Tutorial mode | Model gives step-by-step instructions instead of creating files |
| Lazy/asking | Model asks user for info instead of using tools |
| Fake actions | Model claims it did something but didn't call any tool |
| Shell in text | Model writes bash commands as text instead of calling `run_bash` |
| Duplicates | Model repeats the same tool call with identical arguments |

Each detector sends an escalating nudge (3 levels) with the correct tool format example.

### Self-Learning

- **Few-shot**: Clean runs (0 nudges, 0 errors) are saved to `data/few_shot.jsonl` and injected as examples for similar future tasks
- **DPO**: After 3+ nudges, bad/good response pairs are collected in `data/dpo_pairs.jsonl` for potential fine-tuning

## License

MIT
