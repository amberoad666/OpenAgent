import ast
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from client import LMClient
from config import SYSTEM_PROMPT, MAX_CONTEXT_TOKENS, COMPACT_THRESHOLD, RULES_REMINDER
import executor
from executor import execute_tool
from input_support import CommandLexer, CommandCompleter, NextActionSuggest
from learning import save_few_shot, build_few_shot_block, build_dpo_block, DPOCollector, show_learn_stats
from tools import undo_last, clear_read_cache
from indexer import CodebaseIndex
from ui import (
    show_welcome, show_help, show_error, show_info, show_assistant_message,
    show_yolo_enabled, show_yolo_disabled, show_token_usage,
    show_dpo_prompt, show_dpo_saved, show_cost,
)

console = Console()


class EscInterrupt(Exception):
    """Raised when user presses Esc to abort the current task."""
    pass


def _check_esc():
    """Non-blocking check: if Esc was pressed, raise EscInterrupt."""
    if sys.platform == "win32":
        import msvcrt
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch == '\x1b':
                raise EscInterrupt()
    else:
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                raise EscInterrupt()


# Debug logging — set OPEN_AGENT_DEBUG=1 to enable, then check debug.log
DEBUG = os.environ.get("OPEN_AGENT_DEBUG", "")
if DEBUG:
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("open-agent")
else:
    log = logging.getLogger("open-agent")
    log.addHandler(logging.NullHandler())

KEEP_RECENT = 4


def load_project_memory():
    """Walk up from cwd looking for .openagent files, collect their contents.

    Returns concatenated memory string or empty string.
    """
    memories = []
    current = os.path.abspath(os.getcwd())
    visited = set()
    while current not in visited:
        visited.add(current)
        candidate = os.path.join(current, ".openagent")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    memories.append(f"# Project memory ({candidate})\n{content}")
            except Exception:
                pass
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Reverse so root-level memory comes first
    memories.reverse()
    return "\n\n".join(memories)


def estimate_tokens(messages):
    """Rough token estimate: ~4 chars per token."""
    return sum(len(m.get("content", "")) for m in messages) // 4


class SessionStats:
    """Track token usage and generation stats for /cost command."""

    def __init__(self):
        self.requests = 0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.total_gen_time = 0.0
        self.start_time = time.time()

    def record_request(self, prompt_chars, completion_chars, gen_time):
        self.requests += 1
        self.prompt_chars += prompt_chars
        self.completion_chars += completion_chars
        self.total_gen_time += gen_time

    def summary(self):
        prompt_tok = self.prompt_chars // 4
        completion_tok = self.completion_chars // 4
        total_tok = prompt_tok + completion_tok
        avg_speed = (completion_tok / self.total_gen_time) if self.total_gen_time > 0 else 0
        avg_time = (self.total_gen_time / self.requests) if self.requests > 0 else 0
        session_time = time.time() - self.start_time
        return {
            "requests": self.requests,
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": total_tok,
            "avg_speed": avg_speed,
            "avg_time": avg_time,
            "session_time": session_time,
        }


codebase_index = None
session_stats = None
added_files = {}  # path → content for /add persistence across compaction


def compact_messages(client, messages, current_task=None):
    """Compact conversation history by summarizing old messages.

    Keeps system prompt (messages[0]) and the last KEEP_RECENT messages.
    Everything in between is summarized by the model.
    Pinned /add files (tracked in module-level added_files) are excluded
    from summarization and re-injected after compaction.
    If current_task is provided, it's injected after compaction so the model
    knows what it was doing and continues working.
    Returns the new messages list.
    """
    if len(messages) <= KEEP_RECENT + 1:
        return messages

    system_msg = messages[0]
    middle = messages[1:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]

    # Exclude pinned /add messages from summarization
    if added_files:
        middle = [m for m in middle if not m.get("content", "").startswith("[Pinned:")]

    show_info("Compacting conversation...")
    try:
        summary = client.summarize(middle)
    except Exception as e:
        show_error(f"Compact failed: {e}")
        return messages

    summary_msg = {
        "role": "system",
        "content": f"Previous conversation summary: {summary}",
    }
    reminder_msg = {
        "role": "system",
        "content": RULES_REMINDER,
    }
    result = [system_msg, summary_msg, reminder_msg]

    # Re-inject pinned /add files
    if added_files:
        for path, content in added_files.items():
            result.append({"role": "user", "content": f"[Pinned: {path}]\n{content}"})

    result += recent

    if current_task:
        result.append({
            "role": "user",
            "content": (
                f"Context was compacted. You were working on this task: {current_task}\n"
                "Continue where you left off. Do NOT start over. "
                "Use your tools to check current state and keep going."
            ),
        })

    return result


_TOOL_NAMES = {"write_file", "read_file", "edit_file", "run_bash", "list_files", "search_files", "web_search"}

# Models sometimes invent tool names — map them to real ones
_TOOL_ALIASES = {
    "execute_bash": "run_bash",
    "bash": "run_bash",
    "shell": "run_bash",
    "execute": "run_bash",
    "create_file": "write_file",
    "cat": "read_file",
    "find_files": "list_files",
    "grep": "search_files",
    "search": "search_files",
    "internet_search": "web_search",
    "search_web": "web_search",
    "google": "web_search",
    "ddg": "web_search",
}


def _resolve_tool_name(name):
    """Resolve a tool name, handling aliases."""
    if name in _TOOL_NAMES:
        return name
    return _TOOL_ALIASES.get(name)

# Detects any tag that signals the model is emitting a tool call as text
_ALL_TOOL_NAMES = _TOOL_NAMES | set(_TOOL_ALIASES.keys())
_SUPPRESS_RE = re.compile(
    r'<(?:think|tool_call|execute|function=|parameter=|/?(?:name|arg_name|arg_value))\b'
    r'|<\|tool_call_end\|>'
    r'|```tool_call'
    r'|\b(?:' + '|'.join(_ALL_TOOL_NAMES) + r')\s*\('
)


def _clean_model_output(content, has_tool_calls=False):
    """Strip ALL model artifacts from content text."""
    # <think>...</think>
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
    # <function=...>...</function>
    content = re.sub(r'<function=\w+>.*?</function>', '', content, flags=re.DOTALL)
    content = re.sub(r'<function=\w+>.*$', '', content, flags=re.DOTALL)
    # <execute>...</execute>
    content = re.sub(r'<execute>.*?</execute>', '', content, flags=re.DOTALL)
    content = re.sub(r'<execute>.*$', '', content, flags=re.DOTALL)
    # <tool_call>...</tool_call>
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
    content = re.sub(r'<tool_call>.*$', '', content, flags=re.DOTALL)
    # Orphan XML tags
    content = re.sub(r'</?(?:tool_call|name|arg_name|arg_value|execute|parameter=[^>]*)>', '', content)
    content = re.sub(r'<\|tool_call_end\|>', '', content)
    # If tool calls were parsed, remove duplicate code blocks
    if has_tool_calls:
        content = re.sub(r'```\w*\n.*?```', '', content, flags=re.DOTALL)
    return content.strip()


def _parse_tool_calls_from_text(content):
    """Universal parser: extract tool calls from model text output.

    Handles ALL known formats:
      1. <function=name><parameter=key>value</parameter></function>  (liquid/lfm2)
      2. <tool_call>{"name":..., "arguments":...}</tool_call>       (JSON in XML)
      3. <tool_call><name>f</name><arg_name>k</arg_name><arg_value>v</arg_value></tool_call>
      4. <execute>func(key="val")</execute>                          (Python-style)
      5. bare func(key="val") in text                                (no wrapper)

    Returns (cleaned_content, tool_calls_list).
    """
    tool_calls = []
    call_spans = []

    # Fix common model typos before parsing
    content = re.sub(r'<function=function=', '<function=', content)
    content = re.sub(r'<parameter=parameter=', '<parameter=', content)

    # --- Format 1: <function=name><parameter=key>value</parameter></function> ---
    for m in re.finditer(r'<function=(\w+)>(.*?)</function>', content, re.DOTALL):
        name = m.group(1)
        body = m.group(2)
        args = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', body, re.DOTALL):
            val = pm.group(2)
            # Strip exactly one leading/trailing newline (XML formatting, not code)
            if val.startswith('\n'):
                val = val[1:]
            if val.endswith('\n'):
                val = val[:-1]
            args[pm.group(1)] = val
        resolved = _resolve_tool_name(name)
        if resolved and args:
            tool_calls.append({
                "id": f"tc_{len(tool_calls)}",
                "name": resolved,
                "arguments": json.dumps(args, ensure_ascii=False),
            })
            call_spans.append((m.start(), m.end()))

    # Unclosed <function= at end
    if not tool_calls:
        m = re.search(r'<function=(\w+)>(.*)', content, re.DOTALL)
        if m:
            name = m.group(1)
            body = m.group(2)
            args = {}
            for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', body, re.DOTALL):
                val = pm.group(2)
                if val.startswith('\n'):
                    val = val[1:]
                if val.endswith('\n'):
                    val = val[:-1]
                args[pm.group(1)] = val
            resolved = _resolve_tool_name(name)
            if resolved and args:
                tool_calls.append({
                    "id": f"tc_{len(tool_calls)}",
                    "name": resolved,
                    "arguments": json.dumps(args, ensure_ascii=False),
                })
                call_spans.append((m.start(), m.end()))

    # --- Format 2: <tool_call> JSON </tool_call> ---
    if not tool_calls:
        for m in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
            inner = m.group(1).strip()
            try:
                parsed = json.loads(inner)
                name = parsed.get("name", parsed.get("function", ""))
                args = parsed.get("arguments", parsed.get("parameters", {}))
                resolved = _resolve_tool_name(name)
                if resolved:
                    tool_calls.append({
                        "id": f"tc_{len(tool_calls)}",
                        "name": resolved,
                        "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                    })
                    call_spans.append((m.start(), m.end()))
            except (json.JSONDecodeError, ValueError):
                pass

    # --- Format 3: <tool_call><name>...</name><arg_name>...</arg_name><arg_value>...</arg_value> ---
    if not tool_calls:
        for m in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
            inner = m.group(1)
            nm = re.search(r'<name>(.*?)</name>', inner, re.DOTALL)
            if nm:
                name = nm.group(1).strip()
                args = {}
                for am in re.findall(r'<arg_name>(.*?)</arg_name>\s*<arg_value>(.*?)</arg_value>', inner, re.DOTALL):
                    args[am[0].strip()] = am[1]
                resolved = _resolve_tool_name(name)
                if resolved:
                    tool_calls.append({
                        "id": f"tc_{len(tool_calls)}",
                        "name": resolved,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    })
                    call_spans.append((m.start(), m.end()))

    # --- Format 4: <execute>func(...)</execute> ---
    if not tool_calls:
        for m in re.finditer(r'<execute>(.*?)</execute>', content, re.DOTALL):
            tc = _parse_python_func_call(m.group(1))
            if tc:
                tool_calls.append(tc)
                call_spans.append((m.start(), m.end()))

    # --- Format 5: ```tool_call code blocks with Python-like calls ---
    if not tool_calls:
        for m in re.finditer(r'```tool_call\n(.*?)```', content, re.DOTALL):
            tc = _parse_code_block_tool_call(m.group(1))
            if tc:
                tool_calls.append(tc)
                call_spans.append((m.start(), m.end()))

    # --- Format 6: bare tool_name(...) anywhere in text ---
    if not tool_calls:
        tool_pat = re.compile(r'\b(' + '|'.join(_TOOL_NAMES) + r')\s*\(')
        for m in tool_pat.finditer(content):
            call_text = _extract_balanced_call(content, m.start())
            if call_text:
                tc = _parse_python_func_call(call_text)
                if tc:
                    tool_calls.append(tc)
                    call_spans.append((m.start(), m.start() + len(call_text)))

    if not tool_calls:
        return content, []

    # Remove call spans from content
    cleaned = content
    for start, end in sorted(call_spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = cleaned.strip()
    return cleaned, tool_calls


def _parse_code_block_tool_call(code):
    """Parse a ```tool_call code block containing Python-style code.

    Handles cases where the model defines variables and then calls a tool function.
    E.g.:
        EDITED_CODE = \"\"\"...\"\"\"
        edit_file(path='bot.py', old_text='', new_text=EDITED_CODE)

    Returns a tool_call dict or None.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # Collect variable assignments (name -> string value)
    variables = {}
    call_node = None

    for node in ast.walk(tree):
        # Variable assignments: X = "..." or X = '''...'''
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                variables[target.id] = node.value.value
        # Function calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if _resolve_tool_name(node.func.id):
                call_node = node

    if not call_node:
        return None

    func_name = _resolve_tool_name(call_node.func.id)
    args = {}
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            args[kw.arg] = kw.value.value
        elif isinstance(kw.value, ast.Name) and kw.value.id in variables:
            args[kw.arg] = variables[kw.value.id]

    if args:
        return {
            "id": f"tc_0",
            "name": func_name,
            "arguments": json.dumps(args, ensure_ascii=False),
        }
    return None


def _extract_balanced_call(text, start):
    """Extract func(...) with balanced parentheses, respecting string literals."""
    paren_pos = text.index('(', start)
    depth = 0
    in_str = None
    escape = False
    i = paren_pos
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == '\\':
            escape = True
            i += 1
            continue
        if in_str:
            if ch == in_str:
                in_str = None
        else:
            if ch in ('"', "'"):
                in_str = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1
    return text[start:]


def _parse_python_func_call(text):
    """Parse func(key="value", ...) — returns a tool_call dict or None."""
    text = text.strip()
    text = re.sub(r'<\|tool_call_end\|>\s*$', '', text).strip()

    m = re.match(r'(\w+)\((.*)\)\s*$', text, re.DOTALL)
    if not m:
        return None
    func_name = _resolve_tool_name(m.group(1))
    if not func_name:
        return None
    args_str = m.group(2)

    # Strategy 1: ast.parse
    try:
        tree = ast.parse(f"_({args_str})", mode='eval')
        args = {}
        for kw in tree.body.keywords:
            if isinstance(kw.value, ast.Constant):
                args[kw.arg] = str(kw.value.value)
        if args:
            return {"id": f"tc_0", "name": func_name, "arguments": json.dumps(args, ensure_ascii=False)}
    except Exception:
        pass

    # Strategy 2: positional key="..." parsing
    args = {}
    key_starts = list(re.finditer(r'(\w+)\s*=\s*(["\'])', args_str))
    for i, km in enumerate(key_starts):
        key = km.group(1)
        quote = km.group(2)
        val_start = km.end()
        if i + 1 < len(key_starts):
            next_key_pos = key_starts[i + 1].start()
            segment = args_str[val_start:next_key_pos]
            last_sep = segment.rfind(quote + ',')
            if last_sep < 0:
                last_sep = segment.rfind(quote)
            value = segment[:last_sep] if last_sep >= 0 else segment
        else:
            segment = args_str[val_start:]
            last_q = segment.rfind(quote)
            value = segment[:last_q] if last_q >= 0 else segment
        value = value.replace('\\n', '\n').replace('\\t', '\t')
        value = value.replace('\\"', '"').replace("\\'", "'")
        args[key] = value

    if args:
        return {"id": f"tc_0", "name": func_name, "arguments": json.dumps(args, ensure_ascii=False)}
    return None


class StreamRenderer:
    """Buffers streaming text; renders code blocks with syntax highlighting."""

    def __init__(self):
        self._buffer = ""         # accumulated text not yet flushed
        self._in_code = False     # inside a ``` block
        self._code_lang = ""
        self._code_lines = []
        self._code_title = ""     # last text line before code fence
        self._flushed_pos = 0     # how much of _buffer we already printed

    @staticmethod
    def _apply_inline_bold(text):
        """Replace **bold** with Rich markup [bold]...[/bold]."""
        return re.sub(r'\*\*(.+?)\*\*', r'[bold]\1[/bold]', text)

    def _render_line(self, line):
        """Render a single line with basic markdown formatting."""
        stripped = line.rstrip('\n')

        # Headers: # Header → bold cyan
        m = re.match(r'^(#{1,3})\s+(.+)$', stripped)
        if m:
            console.print(f"[bold cyan]{m.group(2)}[/bold cyan]")
            return

        # Unordered lists: - item or * item
        m = re.match(r'^(\s*)[*-]\s+(.+)$', stripped)
        if m:
            indent = m.group(1)
            text = self._apply_inline_bold(m.group(2))
            console.print(f"{indent}[dim]•[/dim] {text}")
            return

        # Ordered lists: 1. item
        m = re.match(r'^(\s*)(\d+)\.\s+(.+)$', stripped)
        if m:
            indent = m.group(1)
            num = m.group(2)
            text = self._apply_inline_bold(m.group(3))
            console.print(f"{indent}[dim]{num}.[/dim] {text}")
            return

        # Regular text with possible **bold**
        if '**' in stripped:
            text = self._apply_inline_bold(stripped)
            console.print(text)
            return

        # Plain text — zero overhead, write directly
        sys.stdout.write(line)
        sys.stdout.flush()

    def feed(self, text):
        """Feed new streaming delta text."""
        self._buffer += text
        self._process()

    def _process(self):
        """Scan buffer for code fences and flush what we can."""
        from rich.syntax import Syntax as RichSyntax
        from rich.panel import Panel

        buf = self._buffer
        pos = self._flushed_pos

        while pos < len(buf):
            if not self._in_code:
                # Look for opening ```
                fence_idx = buf.find("```", pos)
                if fence_idx == -1:
                    # No fence found — flush everything up to the last newline
                    # (keep partial line in case ``` comes next)
                    last_nl = buf.rfind("\n", pos)
                    if last_nl >= pos:
                        for ln in buf[pos:last_nl + 1].splitlines(True):
                            self._render_line(ln)
                        pos = last_nl + 1
                    break
                else:
                    # Grab last non-empty line before fence as title
                    before = buf[pos:fence_idx]
                    title_lines = [l.strip() for l in before.rstrip().split("\n") if l.strip()]
                    self._code_title = title_lines[-1].rstrip(":") if title_lines else ""
                    # Flush text before the fence
                    if fence_idx > pos:
                        for ln in buf[pos:fence_idx].splitlines(True):
                            self._render_line(ln)
                    # Find end of opening fence line
                    nl = buf.find("\n", fence_idx)
                    if nl == -1:
                        pos = fence_idx
                        break  # wait for more data
                    lang_line = buf[fence_idx + 3:nl].strip()
                    self._code_lang = lang_line if lang_line else "text"
                    self._in_code = True
                    self._code_lines = []
                    pos = nl + 1
            else:
                # Inside code block — look for closing ```
                close_idx = buf.find("\n```", pos)
                if close_idx == -1:
                    # Also check if the line IS just ```
                    close_idx2 = buf.find("```", pos)
                    if close_idx2 != -1 and (close_idx2 == pos or buf[close_idx2 - 1] == "\n"):
                        # Found closing fence
                        code_text = buf[pos:close_idx2]
                        self._code_lines.append(code_text)
                        code = "".join(self._code_lines).rstrip("\n")
                        # Render with syntax highlighting in a panel
                        title = self._code_title or self._code_lang
                        try:
                            syn = RichSyntax(code, self._code_lang, theme="monokai", line_numbers=True)
                            console.print(Panel(syn, title=title, border_style="cyan", padding=(0, 1)))
                        except Exception:
                            sys.stdout.write(f"```{self._code_lang}\n{code}\n```\n")
                            sys.stdout.flush()
                        self._in_code = False
                        # Skip past closing ```
                        end = buf.find("\n", close_idx2)
                        pos = (end + 1) if end != -1 else close_idx2 + 3
                    else:
                        # No closing fence yet — buffer code
                        self._code_lines.append(buf[pos:])
                        pos = len(buf)
                        break
                else:
                    # Closing fence found
                    code_text = buf[pos:close_idx]
                    self._code_lines.append(code_text)
                    code = "".join(self._code_lines).rstrip("\n")
                    title = self._code_title or self._code_lang
                    try:
                        syn = RichSyntax(code, self._code_lang, theme="monokai", line_numbers=True)
                        console.print(Panel(syn, title=title, border_style="cyan", padding=(0, 1)))
                    except Exception:
                        sys.stdout.write(f"```{self._code_lang}\n{code}\n```\n")
                        sys.stdout.flush()
                    self._in_code = False
                    # Skip past closing ``` line
                    end = buf.find("\n", close_idx + 1)
                    pos = (end + 1) if end != -1 else close_idx + 4

        self._flushed_pos = pos

    def trim_tool_leak(self):
        """Remove any unflushed partial tool-call tag from the buffer.

        Called when stream suppression triggers — the trailing '<func...'
        that was fed before the full '<function=' was detected should not
        be printed.
        """
        buf = self._buffer
        pos = self._flushed_pos
        if pos >= len(buf):
            return
        remaining = buf[pos:]
        idx = remaining.rfind('<')
        if idx >= 0:
            # Truncate: everything from that '<' onward is tool-call garbage
            self._buffer = buf[:pos + idx]

    def finish(self):
        """Flush any remaining buffered content."""
        buf = self._buffer
        pos = self._flushed_pos
        if self._in_code:
            # Unclosed code block — just print raw
            code = "".join(self._code_lines) + buf[pos:]
            sys.stdout.write(code)
            sys.stdout.flush()
        elif pos < len(buf):
            for ln in buf[pos:].splitlines(True):
                self._render_line(ln)


def process_stream(stream):
    """Process a streaming response, collecting text and tool calls.

    We do NOT use the API tools parameter — models break with it.
    Instead, models emit tool calls as text and we parse them.

    Returns (content_text, tool_calls_list).
    """
    content = ""
    suppressed = False  # once True, stop printing to stdout
    renderer = StreamRenderer()
    ever_rendered = False  # did renderer get any text before suppression?

    for chunk in stream:
        _check_esc()  # let user abort mid-stream with Esc
        if DEBUG:
            log.debug("stream chunk: %s", json.dumps(chunk, ensure_ascii=False))
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        if delta.get("content"):
            content += delta["content"]
            if not suppressed:
                if _SUPPRESS_RE.search(content):
                    suppressed = True
                    renderer.trim_tool_leak()
                else:
                    renderer.feed(delta["content"])
                    ever_rendered = True

    # Flush renderer if it got any text
    if ever_rendered:
        renderer.finish()

    if DEBUG:
        log.debug("stream raw content (%d chars): %s", len(content), content[:1000])

    # Parse tool calls from the text
    tool_calls = []
    if content:
        cleaned, tool_calls = _parse_tool_calls_from_text(content)
        if tool_calls:
            if DEBUG:
                log.debug("parsed tool calls: %s", json.dumps(tool_calls, ensure_ascii=False))
            content = cleaned

    # Clean remaining artifacts
    content = _clean_model_output(content, has_tool_calls=bool(tool_calls))

    # Show clean text (only if renderer didn't already print it)
    if suppressed and content and not ever_rendered:
        # Model output was entirely tool calls — print the cleaned text
        sys.stdout.write(content + "\n")
        sys.stdout.flush()
    elif content:
        # Renderer already printed the text — just add newline
        sys.stdout.write("\n")

    if DEBUG:
        log.debug("stream done — content: %s", content[:500] if content else "(empty)")
        log.debug("stream done — tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False))

    return content, tool_calls


# Phrases that signal the model is asking/refusing instead of using tools
_LAZY_PATTERNS = re.compile(
    r'(?:please\s+(?:share|provide|paste|send|show|give)|'
    r'could\s+you\s+(?:share|provide|paste|send|show|give)|'
    r'can\s+you\s+(?:share|provide|paste|send|show)|'
    r'share\s+the\s+code|'
    r'provide\s+(?:the|your)\s+(?:code|file|content|error)|'
    r'paste\s+(?:the|your)\s+(?:code|file|content)|'
    r'I\s+need\s+(?:the|your)\s+(?:code|file|content)|'
    r'what\s+(?:is|are)\s+the\s+(?:error|issue|problem)|'
    r"I\s+(?:don't|do\s+not|cannot|can't)\s+have\s+(?:direct\s+)?access|"
    r"I\s+(?:don't|do\s+not|cannot|can't)\s+(?:directly\s+)?(?:access|check|see|view|run|test|execute)|"
    r"don't\s+have\s+access\s+to\s+your|"
    r"no\s+(?:direct\s+)?access\s+to|"
    # English excuses about testing/running
    r"I\s+(?:can't|cannot|couldn't|could\s+not)\s+(?:run|test|execute|launch|start)|"
    r"(?:requires|needs)\s+(?:a\s+)?(?:token|key|credentials|environment|setup)\s+(?:to|for)|"
    r"(?:ready|able)\s+to\s+(?:provide|send|share|show)|"
    r"on\s+your\s+(?:computer|machine|system|end)|"
    # Russian: model asks user to provide/show/send files
    r"(?:пожалуйста[,.]?\s+)?предоставьте\s+(?:файл|код|содержимое|ошибку)|"
    r"(?:пожалуйста[,.]?\s+)?(?:покажите|пришлите|отправьте|скиньте)\s+(?:файл|код|содержимое)|"
    r"мне\s+нужно\s+(?:сначала\s+)?(?:увидеть|посмотреть|прочитать)|"
    r"(?:нужно|необходимо)\s+(?:сначала\s+)?(?:увидеть|посмотреть|прочитать)\s+(?:его|её|код|файл)|"
    r"чтобы\s+(?:исправить|помочь|проверить).{0,40}(?:предоставьте|покажите|нужно\s+увидеть)|"
    # Russian excuses
    r"у\s+меня\s+нет\s+доступа|"
    r"не\s+(?:имею|могу)\s+(?:доступ|проверить)|"
    r"не\s+мог(?:у|ла?)?\s+(?:провести|запустить|выполнить|протестировать)|"
    r"готов\s+(?:предоставить|отправить|показать)|"
    r"могу\s+(?:предоставить|отправить|показать)|"
    r"требуется\s+(?:токен|ключ|окружение|настройка)|"
    r"для\s+запуска\s+(?:требуется|нужен|необходим)|"
    r"на\s+вашем\s+компьютере|"
    r"(?:если|когда)\s+(?:нужно|хотите).*(?:могу|готов)|"
    # Russian: model asks for details/info instead of working
    r"мне\s+нужн[оыа]\s+(?:знать\s+)?(?:некоторые\s+|несколько\s+)?(?:детал|подробност|информаци|данн|уточнени)|"
    r"(?:укажите|уточните|подскажите|сообщите|расскажите)\s+(?:пожалуйста\s+)?(?:какой|какие|какое|что|как|где|сколько)|"
    r"для\s+(?:создания|разработки|продолжения|работы|реализации).{0,40}(?:нужн[оыа]|необходим[оыа]|требуется)|"
    r"(?:какой|какие|какое)\s+(?:именно|конкретно)\s+(?:дизайн|стиль|цвет|тем[ау]|шрифт|контент|товар)|"
    r"(?:есть\s+ли|имеются\s+ли)\s+(?:у\s+вас\s+)?(?:предпочтения|пожелания|требования|логотип|данные)|"
    r"(?:пожалуйста|прошу).*(?:уточн|предоставь|сообщ|расскаж|поделитесь))",
    re.IGNORECASE,
)


def _is_asking_instead_of_acting(content):
    """Detect if the model is asking the user for info instead of using tools."""
    return bool(_LAZY_PATTERNS.search(content))


# Phrases that signal the model CLAIMS it did something but didn't call a tool
_FAKE_ACTION_PATTERNS = re.compile(
    r'(?:исправлен|заменен|обновлен|изменен|исправил|заменил|обновил|изменил|'
    r'импорт(?:ы)?\s+(?:исправлен|заменен|обновлен)|'
    r'исправление\s+(?:выполнено|завершено|сделано)|'
    r'код\s+(?:исправлен|обновлен|заменен)|'
    r'файл\s+(?:исправлен|обновлен|изменен)|'
    # Fake testing/running claims (RU)
    r'(?:я\s+)?выполнил\w*\s+(?:тестовый\s+)?(?:запуск|проверку|тест)|'
    r'(?:я\s+)?(?:проверил|протестировал|запустил|провер(?:ил|ял))\w*\s+(?:его|её|бот|код|скрипт|файл|работу|запуск)|'
    r'(?:бот|скрипт|код|программа)\s+(?:запущен|работает|запускается|корректно)|'
    r'(?:бот|скрипт|код|программа)\s+.{0,80}(?:запущен|работает|запускается|корректно)|'
    r'запускается\s+(?:без\s+ошибок|с\s+указанным|с\s+токеном|корректно|успешно)|'
    r'готов\s+к\s+использованию|'
    r'(?:теперь|сейчас)\s+(?:проверю|запущу|протестирую|проверим)|'
    # Model narrates file creation without actually calling tools
    # Present: создаю, Future: создам, Past: создал
    r'(?:я\s+)?(?:создаю|создам|создал[аи]?|напишу|напишем)\s+(?:файл|первый\s+файл|следующий\s+файл)|'
    r'(?:я\s+)?(?:добавляю|добавлю|добавил[аи]?)\s+(?:маршрут|обработку|обработчик|данные|код|настройку|конфигурацию)|'
    r'(?:затем|далее|теперь|сейчас)\s+(?:добавляю|добавлю|создаю|создам|настраиваю|обновляю|обновлю|напишу)|'
    r'(?:задача|работа)\s+завершена|'
    # Model says "I'll execute command" but shows code block instead of calling tool
    r'(?:я\s+)?выполню\s+команду|'
    r'(?:я\s+)?(?:остановлю|завершу|убью)\s+(?:процесс|бот|сервер|скрипт)|'
    r'для\s+остановки\s+(?:процесса|бота|сервера)\s+используйте|'
    r'(?:I\s+)?(?:fixed|replaced|updated|changed|modified)\s+(?:the\s+)?(?:code|file|import|line)|'
    # Fake testing/running claims (EN)
    r'(?:I\s+)?(?:tested|ran|launched|executed|started|verified)\s+(?:the\s+)?(?:code|bot|script|file|program)|'
    # Model suggests user run a command manually instead of using run_bash
    r'(?:используйте|выполните|запустите)\s+команду|'
    r'(?:use|run|execute)\s+(?:the\s+)?(?:command|following))',
    re.IGNORECASE,
)


def _is_fake_action(content):
    """Detect if model claims it made changes but didn't use any tool."""
    return bool(_FAKE_ACTION_PATTERNS.search(content))


# Detect shell commands in text — model writes bash instead of calling tools
_SHELL_IN_TEXT_PATTERNS = re.compile(
    r'(?:'
    # echo/cat redirecting to file
    r'echo\s+["\'].*?["\']?\s*>{1,2}\s*\S+|'
    r'cat\s+(?:<<|>{1,2})\s*\S+|'
    # file operations as shell commands
    r'mkdir\s+-?p?\s+\S+|'
    r'touch\s+\S+\.\w+|'
    r'sed\s+-[ie]\s|'
    # wrong tool names (from other AI systems)
    r'str_replace_editor|'
    r'create_file\s*\(|'
    r'execute_bash\s*\(|'
    # bash code blocks with file-writing commands
    r'```(?:bash|sh|shell|zsh)\s*\n\s*(?:echo|cat|mkdir|touch|sed|printf)\s|'
    # Numbered tutorial steps: "1. Создайте файл" / "1. Create file"
    r'\d+\.\s*(?:Создайте|Добавьте|Create|Add)\s+(?:файл|папку|file|folder|directory|the\s+following)|'
    # Code blocks with filename header: "file.js:" or "`file.js`:" followed by code block
    r'`[^`]+\.(?:js|ts|py|json|html|css|jsx|tsx)`\s*:\s*\n\s*```'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


def _is_shell_in_text(content):
    """Detect if model outputs shell commands as text instead of calling tools."""
    return bool(_SHELL_IN_TEXT_PATTERNS.search(content))


# Tutorial/instruction mode: model gives step-by-step instructions with code blocks
# instead of using write_file to create files itself
_TUTORIAL_PATTERNS = re.compile(
    r'(?:'
    # Numbered steps: "1. Создайте файл" / "1. Create file" / "4. Добавьте данные в X"
    r'\d+\.\s*(?:Создайте|Измените|Обновите|Create|Update|Modify)\s+(?:файл|папку|директорию|каталог|file|folder|directory)|'
    r'\d+\.\s*(?:Добавьте|Add)\s+(?:файл|папку|данные|код|содержимое|file|folder|data|code|content|the\s+following)|'
    # Imperative with backtick filename: "Создайте файл `server.js`:"
    r'(?:Создайте|Добавьте|Create|Add)\s+(?:файл|file)\s+`[^`]+`|'
    # "Добавьте данные/код/содержимое в `file`:"
    r'(?:Добавьте|Add)\s+(?:данные|код|содержимое|the\s+following)\s+(?:в|to)\s+`[^`]+`|'
    # Code block right after filename header: "`server.js`:" + newline + ```
    r'`[^`]+\.(?:js|ts|py|json|html|css|jsx|tsx|yaml|toml|cfg|ini)`\s*:\s*\n\s*```'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


def _is_tutorial_mode(content):
    """Detect if model gives step-by-step tutorial instead of using tools."""
    return bool(_TUTORIAL_PATTERNS.search(content))


MAX_TOOL_ROUNDS = 200  # practically unlimited — let the model finish


def _find_user_task(messages):
    """Extract the current user task from messages (last user message before tools)."""
    for msg in reversed(messages):
        if msg["role"] == "user" and not msg["content"].startswith("[Tool result:"):
            return msg["content"]
    return None


def _generate_suggestion(client, messages, response_text):
    """Quick non-streaming model call to suggest the next user action."""
    try:
        last_user = _find_user_task(messages) or ""
        summary = response_text[:200] if response_text else ""
        suggestion_msgs = [
            {"role": "system", "content": (
                "Based on the conversation, suggest ONE short next action "
                "the user might want to do. Reply in the user's language. "
                "Just the action, 3-10 words, no explanation, no quotes."
            )},
            {"role": "user", "content": last_user[:100]},
            {"role": "assistant", "content": summary},
            {"role": "user", "content": "What should I do next?"},
        ]
        result = client.chat(suggestion_msgs)
        text = result.get("content", "").strip()
        text = text.strip('"\'')
        return text[:80] if text else ""
    except Exception:
        return ""


def run_conversation_turn(client, messages, max_tokens=None):
    """Run one turn of conversation, handling tool calls in a loop.

    We don't pass tools to the API — models can't use them reliably.
    Instead we parse tool calls from the model's text output.

    Returns the final assistant text response.
    """
    clear_read_cache()
    rounds = 0
    errors_in_row = 0
    duplicate_nudges = 0  # how many times we nudged for repeats
    fake_nudges = 0       # how many times we nudged for fake actions
    lazy_nudges = 0       # how many times we nudged for laziness
    tutorial_nudges = 0   # how many times we nudged for tutorial/instruction mode
    user_rejections = 0   # how many times user rejected tool calls
    prev_calls = set()    # track (name, args_json) to detect duplicates
    edit_failures = {}  # path → consecutive failure count
    write_paths = {}    # path → count of write_file calls (detect rewrites)

    # Learning: track tool chain and nudges for few-shot / DPO
    tool_chain = []
    total_nudges = 0
    dpo_collector = None
    training_mode = False
    turn_start_idx = len(messages)
    user_task = _find_user_task(messages)

    while True:
        _check_esc()  # let user abort with Esc between rounds

        # Auto-compact mid-turn if context is growing too large
        if max_tokens and estimate_tokens(messages) > max_tokens * COMPACT_THRESHOLD:
            task = _find_user_task(messages)
            messages[:] = compact_messages(client, messages, current_task=task)

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        t0 = time.time()
        stream = client.stream_chat(messages)
        content, tool_calls = process_stream(stream)
        gen_time = time.time() - t0
        if session_stats:
            session_stats.record_request(prompt_chars, len(content or ""), gen_time)

        # Add assistant message to history
        assistant_msg = {"role": "assistant", "content": content or ""}
        messages.append(assistant_msg)

        if DEBUG:
            log.debug("assistant_msg: %s", json.dumps(assistant_msg, ensure_ascii=False)[:500])

        if not tool_calls:
            # Detect tutorial/instruction mode — model gives steps + code blocks
            # instead of calling write_file itself. Check this FIRST (most specific).
            if content and _is_tutorial_mode(content) and tutorial_nudges < 3:
                tutorial_nudges += 1
                if tutorial_nudges == 1:
                    nudge = (
                        "WRONG. You gave INSTRUCTIONS instead of DOING THE WORK.\n"
                        "The user asked YOU to create these files. Do NOT tell the user what to create.\n"
                        "You MUST call write_file for EACH file. Like this:\n\n"
                        "<function=write_file>\n"
                        "<parameter=path>server.js</parameter>\n"
                        "<parameter=content>\n"
                        "const express = require('express');\n"
                        "const app = express();\n"
                        "app.listen(3000);\n"
                        "</parameter>\n"
                        "</function>\n\n"
                        "Create the FIRST file NOW using <function=write_file>. One file at a time."
                    )
                elif tutorial_nudges == 2:
                    nudge = (
                        "STOP giving instructions. You wrote code blocks as TEXT — that creates NOTHING.\n"
                        "The ONLY way to create a file is:\n"
                        "<function=write_file>\n"
                        "<parameter=path>FILENAME</parameter>\n"
                        "<parameter=content>\nFILE CONTENT HERE\n</parameter>\n"
                        "</function>\n\n"
                        "Do NOT write numbered steps. Do NOT show code blocks.\n"
                        "Call <function=write_file> RIGHT NOW with the first file."
                    )
                else:
                    nudge = (
                        "FINAL WARNING. You keep showing code instead of creating files.\n"
                        "Call write_file NOW or the task will be aborted:\n"
                        "<function=write_file>\n"
                        "<parameter=path>index.html</parameter>\n"
                        "<parameter=content>\n<!DOCTYPE html><html><body>Hello</body></html>\n</parameter>\n"
                        "</function>"
                    )
                messages.append({"role": "user", "content": nudge})
                total_nudges += 1
                if total_nudges >= 3 and not training_mode:
                    if show_dpo_prompt():
                        dpo_collector = DPOCollector(user_task, messages[0]["content"])
                        dpo_collector.activate()
                    training_mode = True
                if dpo_collector and dpo_collector.active:
                    dpo_collector.add_rejected(messages[-2:])
                rounds += 1
                continue
            # Detect lazy responses — model asks/refuses instead of acting
            if content and _is_asking_instead_of_acting(content) and lazy_nudges < 3:
                lazy_nudges += 1
                if lazy_nudges == 1:
                    nudge = (
                        "WRONG. Do NOT ask the user for details. You are AUTONOMOUS.\n"
                        "- Missing info (name, style, content)? Make a reasonable choice yourself.\n"
                        "- Don't know what files exist? Call list_files RIGHT NOW.\n"
                        "- Need to check existing code? Call read_file.\n"
                        "Step 1: list_files() to see what's already in the directory.\n"
                        "Step 2: read_file(path) for any relevant files.\n"
                        "Step 3: Do the task. If details are missing, INVENT them.\n"
                        "Act NOW. Do NOT reply with text."
                    )
                elif lazy_nudges == 2:
                    nudge = (
                        "STOP ASKING. You said the same thing again. The user TOLD you to find the files yourself. "
                        "You MUST call a tool RIGHT NOW. Here is what you do:\n"
                        "<function=list_files>\n<parameter=directory>.</parameter>\n<parameter=pattern>*.py</parameter>\n</function>\n"
                        "Then read the files you find. Do NOT reply with text. CALL THE TOOL."
                    )
                else:
                    nudge = (
                        "FINAL WARNING. Call list_files NOW or the task will be aborted. "
                        "Do NOT write any text. Just call:\n"
                        "<function=list_files>\n<parameter=directory>.</parameter>\n</function>"
                    )
                messages.append({"role": "user", "content": nudge})
                total_nudges += 1
                if total_nudges >= 3 and not training_mode:
                    if show_dpo_prompt():
                        dpo_collector = DPOCollector(user_task, messages[0]["content"])
                        dpo_collector.activate()
                    training_mode = True
                if dpo_collector and dpo_collector.active:
                    dpo_collector.add_rejected(messages[-2:])
                rounds += 1
                continue
            # Detect fake actions — model claims it did something but didn't call any tool
            if content and _is_fake_action(content) and fake_nudges < 3:
                fake_nudges += 1
                nudge = (
                    "STOP. You SAID you would do something but you did NOT call any tool. "
                    "Nothing actually happened. Saying 'создам файл' does NOTHING.\n"
                    "You MUST call write_file NOW. Example:\n\n"
                    "<function=write_file>\n"
                    "<parameter=path>data/products.json</parameter>\n"
                    "<parameter=content>\n"
                    '[{"id": 1, "name": "Product", "price": 100}]\n'
                    "</parameter>\n"
                    "</function>\n\n"
                    "Do NOT narrate. Do NOT say 'я создам/создаю'. CALL the tool NOW."
                )
                messages.append({"role": "user", "content": nudge})
                total_nudges += 1
                if total_nudges >= 3 and not training_mode:
                    if show_dpo_prompt():
                        dpo_collector = DPOCollector(user_task, messages[0]["content"])
                        dpo_collector.activate()
                    training_mode = True
                if dpo_collector and dpo_collector.active:
                    dpo_collector.add_rejected(messages[-2:])
                rounds += 1
                continue
            # Detect shell commands in text — model writes bash instead of calling tools
            if content and _is_shell_in_text(content) and fake_nudges < 3:
                fake_nudges += 1
                nudge = (
                    "WRONG FORMAT. You wrote shell commands as plain text. That does NOTHING.\n"
                    "You MUST use the tool call format. Examples:\n\n"
                    "To create/write a file:\n"
                    "<function=write_file>\n"
                    "<parameter=path>filename.html</parameter>\n"
                    "<parameter=content>\nYour file content here\n</parameter>\n"
                    "</function>\n\n"
                    "To run a command:\n"
                    "<function=run_bash>\n"
                    "<parameter=command>python script.py</parameter>\n"
                    "</function>\n\n"
                    "To edit a file:\n"
                    "<function=edit_file>\n"
                    "<parameter=path>file.py</parameter>\n"
                    "<parameter=old_text>old code</parameter>\n"
                    "<parameter=new_text>new code</parameter>\n"
                    "</function>\n\n"
                    "Do NOT write echo, cat, mkdir, sed, or any shell commands as text. "
                    "Use the <function=...> format above. Do it NOW."
                )
                messages.append({"role": "user", "content": nudge})
                total_nudges += 1
                if total_nudges >= 3 and not training_mode:
                    if show_dpo_prompt():
                        dpo_collector = DPOCollector(user_task, messages[0]["content"])
                        dpo_collector.activate()
                    training_mode = True
                if dpo_collector and dpo_collector.active:
                    dpo_collector.add_rejected(messages[-2:])
                rounds += 1
                continue
            # Learning: save successful few-shot if clean run
            if tool_chain and total_nudges == 0 and errors_in_row == 0:
                save_few_shot(user_task, tool_chain, content, rounds)
            # Learning: save DPO chosen if collector active
            if dpo_collector and dpo_collector.active:
                dpo_collector.set_chosen(messages[turn_start_idx:])
                saved = dpo_collector.save()
                if saved:
                    total = DPOCollector.count_pairs()
                    show_dpo_saved(saved, total)
            return content

        rounds += 1

        # Detect duplicate tool calls (same name + same args as previous round)
        current_calls = {(tc["name"], tc["arguments"]) for tc in tool_calls}
        if current_calls & prev_calls:
            duplicate_nudges += 1
            if duplicate_nudges >= 3:
                show_error("Model keeps repeating despite nudges. Stopping.")
                return content
            repeated = ", ".join(tc["name"] for tc in tool_calls)
            nudge = (
                f"STOP. You already called {repeated} with the same arguments and got the same result. "
                "This approach is not working. Try a DIFFERENT strategy:\n"
                "- If edit_file failed, use read_file first to see the actual file content, then retry with correct old_text.\n"
                "- If run_bash failed, check the error and fix the root cause.\n"
                "- If a file is not found, use list_files to find the correct path.\n"
                "- If you don't understand the error, use web_search to find the solution.\n"
                "Think step by step about what went wrong and try a new approach."
            )
            messages.append({"role": "user", "content": nudge})
            total_nudges += 1
            if total_nudges >= 3 and not training_mode:
                if show_dpo_prompt():
                    dpo_collector = DPOCollector(user_task, messages[0]["content"])
                    dpo_collector.activate()
                training_mode = True
            if dpo_collector and dpo_collector.active:
                dpo_collector.add_rejected(messages[-2:])
            prev_calls = set()
            continue
        prev_calls = current_calls
        duplicate_nudges = 0

        # Stop if too many rounds
        if rounds > MAX_TOOL_ROUNDS:
            show_error(f"Too many tool rounds ({MAX_TOOL_ROUNDS}). Stopping.")
            return content

        # Stop after 2 consecutive errors
        if errors_in_row >= 2:
            show_error("Multiple errors in a row. Stopping.")
            return content

        if content:
            console.print()

        # Execute tool calls — safe tools in parallel, dangerous sequentially
        round_had_error = False
        from executor import is_safe_tool

        # Results collector: index → (name, args, result, is_error, is_rejection)
        tool_results = {}

        # Pre-check: block excessive rewrites of the same file
        for i, tc in enumerate(tool_calls):
            if tc["name"] == "write_file":
                try:
                    a = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    a = {}
                wpath = a.get("path", "")
                if wpath and write_paths.get(wpath, 0) >= 3:
                    # Block write — tell model to move on
                    block_msg = (
                        f"BLOCKED: You already wrote '{wpath}' {write_paths[wpath]} times! "
                        f"This file exists and is fine. STOP rewriting it.\n"
                        f"Move on to the NEXT file you haven't created yet. "
                        f"Files created so far: {', '.join(write_paths.keys())}"
                    )
                    tool_results[i] = (tc["name"], a, block_msg, True, False)
                    show_info(f"Blocked rewrite #{write_paths[wpath]+1} of {wpath}")
                    write_paths[wpath] += 1
                    continue

        # Separate safe and dangerous calls, preserving order indices
        safe_indices = []
        dangerous_indices = []
        for i, tc in enumerate(tool_calls):
            if i in tool_results:
                continue  # already handled (blocked rewrite)
            name = tc["name"]
            if is_safe_tool(name) and not (name in {"read_file"} and executor.skip_permissions is False and False):
                safe_indices.append(i)
            else:
                dangerous_indices.append(i)

        # Run safe tools in parallel
        if len(safe_indices) > 1:
            def _exec_safe(idx):
                tc = tool_calls[idx]
                n = tc["name"]
                try:
                    a = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    a = {}
                r = execute_tool(n, a)
                return idx, n, a, r

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_exec_safe, i): i for i in safe_indices}
                for future in concurrent.futures.as_completed(futures):
                    idx, n, a, r = future.result()
                    is_err = r and (r.startswith("Error:") or r.startswith("Action cancelled") or "not found" in r)
                    is_rej = r and ("User rejected" in r or r.startswith("Action cancelled"))
                    tool_results[idx] = (n, a, r, is_err, is_rej)
        else:
            # Single safe tool or none — run sequentially
            for idx in safe_indices:
                tc = tool_calls[idx]
                n = tc["name"]
                try:
                    a = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    a = {}
                r = execute_tool(n, a)
                is_err = r and (r.startswith("Error:") or r.startswith("Action cancelled") or "not found" in r)
                is_rej = r and ("User rejected" in r or r.startswith("Action cancelled"))
                tool_results[idx] = (n, a, r, is_err, is_rej)

        # Run dangerous tools sequentially
        for idx in dangerous_indices:
            tc = tool_calls[idx]
            n = tc["name"]
            try:
                a = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                a = {}
            r = execute_tool(n, a)
            is_err = r and (r.startswith("Error:") or r.startswith("Action cancelled") or "not found" in r)
            is_rej = r and ("User rejected" in r or r.startswith("Action cancelled"))
            tool_results[idx] = (n, a, r, is_err, is_rej)

        # Process results in original order
        for i in range(len(tool_calls)):
            name, args, result, is_error, is_rejection = tool_results[i]

            if DEBUG:
                log.debug("tool %s(%s) => %s", name, json.dumps(args, ensure_ascii=False), result[:500] if result else "(empty)")

            if is_rejection:
                user_rejections += 1

            if is_error:
                round_had_error = True

            # Auto-retry: track edit_file failures
            if name == "edit_file":
                edit_path = args.get("path", "")
                if is_error:
                    edit_failures[edit_path] = edit_failures.get(edit_path, 0) + 1
                    if edit_failures[edit_path] >= 3:
                        from tools import read_file as _read_file
                        file_content = _read_file(edit_path)
                        show_info(f"Auto-reading {edit_path} after {edit_failures[edit_path]} failed edits")
                        messages.append({
                            "role": "user",
                            "content": f"[Auto-read: {edit_path} (after {edit_failures[edit_path]} failed edits)]\n{file_content}",
                        })
                        edit_failures[edit_path] = 0
                else:
                    edit_failures.pop(edit_path, None)

            # Mark codebase index stale after file modifications
            if name in ("write_file", "edit_file") and not is_error and codebase_index:
                codebase_index.mark_stale()

            # Track write_file paths to detect rewrites
            rewrite_warning = ""
            if name == "write_file" and not is_error:
                wpath = args.get("path", "")
                write_paths[wpath] = write_paths.get(wpath, 0) + 1
                if write_paths[wpath] == 2:
                    rewrite_warning = (
                        f"\nWARNING: You already wrote '{wpath}' — do NOT rewrite it again! "
                        "Move to the NEXT file you haven't created yet."
                    )
                elif write_paths[wpath] >= 3:
                    rewrite_warning = (
                        f"\nSTOP! '{wpath}' has been written {write_paths[wpath]} times! "
                        "Further writes to this file will be BLOCKED. "
                        f"Files so far: {', '.join(write_paths.keys())}. "
                        "Create a NEW file NOW."
                    )

            # Learning: track tool execution
            tool_chain.append({
                "tool": name,
                "args": {k: str(v)[:80] for k, v in args.items()},
                "success": not is_error,
            })

            # Smarter reminder based on result
            if is_rejection:
                reminder = (
                    "The USER declined this action. Read their feedback above carefully. "
                    "Follow their instruction and continue working on the original task. "
                    "Do NOT retry the declined action. Adjust your approach."
                )
            elif is_error:
                reminder = (
                    "The action failed. Read the error above carefully. "
                    "If you need to fix code, use read_file first, then edit_file. "
                    "If you don't know how to fix the error, use web_search to find the solution. "
                    "Use <function=tool_name> format."
                )
            elif name == "run_bash":
                reminder = (
                    "If the task is complete, reply with a short summary. "
                    "Do NOT call more tools unless there is more work to do. "
                    "IMPORTANT: if a command failed with 'No such file', check the "
                    "actual file name and working directory — do NOT guess."
                )
            elif name == "write_file":
                reminder = (
                    "File created. Now create the NEXT file if there are more to create. "
                    "Do NOT rewrite the file you just created. Move forward."
                    + rewrite_warning
                )
            else:
                reminder = (
                    "If the task is complete, reply with a short summary. "
                    "Do NOT call more tools unless there is more work to do."
                )

            messages.append({
                "role": "user",
                "content": f"[Tool result: {name}]\n{result}\n\n{reminder}",
            })

        if round_had_error:
            errors_in_row += 1
        else:
            errors_in_row = 0

        # User rejections → trigger DPO (explicit correction signal)
        if user_rejections >= 3 and not training_mode:
            if show_dpo_prompt("Вы отклонили 3+ действия модели."):
                dpo_collector = DPOCollector(user_task, messages[0]["content"])
                dpo_collector.activate()
            training_mode = True
        if dpo_collector and dpo_collector.active and is_rejection:
            dpo_collector.add_rejected(messages[-2:])

        # Loop to let the model process tool results


def extract_code_blocks(text):
    """Extract fenced code blocks from markdown text.
    Returns list of (language, code) tuples.
    """
    pattern = r"```(\w*)\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)



def save_code_from_response(last_response):
    """Extract code blocks from last response and let user save them."""
    if not last_response:
        show_error("No previous response to save from.")
        return

    blocks = extract_code_blocks(last_response)
    if not blocks:
        show_error("No code blocks found in last response.")
        return

    if len(blocks) == 1:
        lang, code = blocks[0]
        lang_label = f" ({lang})" if lang else ""
        console.print(f"\n[bold]Found 1 code block{lang_label}[/]")
        try:
            path = input("Save to file: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not path:
            return
    else:
        console.print(f"\n[bold]Found {len(blocks)} code blocks:[/]")
        for i, (lang, code) in enumerate(blocks, 1):
            preview = code.strip().split("\n")[0][:60]
            lang_label = f" [{lang}]" if lang else ""
            console.print(f"  [cyan]{i}[/]){lang_label} {preview}...")
        try:
            choice = input(f"\nWhich block? [1-{len(blocks)}, or 'all']: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice.lower() == "all":
            try:
                path = input("Save all to file: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not path:
                return
            code = "\n\n".join(c.strip() for _, c in blocks)
            lang = blocks[0][0]
        elif choice.isdigit() and 1 <= int(choice) <= len(blocks):
            lang, code = blocks[int(choice) - 1]
            try:
                path = input("Save to file: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not path:
                return
        else:
            show_error("Invalid choice.")
            return

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code.strip() + "\n")
        show_info(f"Saved to {path}")
    except Exception as e:
        show_error(f"Failed to save: {e}")


def _summarize_file(fpath):
    """Smart file summary: full content for small files, structure for large ones."""
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading {fpath}: {e}"

    if len(lines) <= 100:
        return "".join(lines)

    # For large files: show structure + head/tail
    summary_parts = [f"[{len(lines)} lines]"]

    # Python files: extract structure with ast
    if fpath.endswith(".py"):
        try:
            source = "".join(lines)
            tree = ast.parse(source)
            structures = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    names = ", ".join(a.name for a in node.names)
                    structures.append(f"  import {names}  (line {node.lineno})")
                elif isinstance(node, ast.ImportFrom):
                    names = ", ".join(a.name for a in node.names)
                    structures.append(f"  from {node.module} import {names}  (line {node.lineno})")
                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    methods_str = f" [{', '.join(methods)}]" if methods else ""
                    structures.append(f"  class {node.name}{methods_str}  (line {node.lineno})")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    structures.append(f"  def {node.name}()  (line {node.lineno})")
            if structures:
                summary_parts.append("Structure:\n" + "\n".join(structures))
        except SyntaxError:
            pass

    # Head + tail
    head = "".join(lines[:10])
    tail = "".join(lines[-5:])
    summary_parts.append(f"First 10 lines:\n{head}")
    summary_parts.append(f"Last 5 lines:\n{tail}")

    return "\n".join(summary_parts)


def _dir_tree(dirpath, max_depth=3, max_files=50):
    """Build a directory tree string."""
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
    result = []
    count = [0]

    def _walk(path, prefix, depth):
        if depth > max_depth or count[0] >= max_files:
            return
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return
        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and e not in skip_dirs]
        files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

        for f in files:
            if count[0] >= max_files:
                result.append(f"{prefix}... (truncated)")
                return
            result.append(f"{prefix}{f}")
            count[0] += 1

        for d in dirs:
            if count[0] >= max_files:
                result.append(f"{prefix}... (truncated)")
                return
            result.append(f"{prefix}{d}/")
            count[0] += 1
            _walk(os.path.join(path, d), prefix + "  ", depth + 1)

    result.append(f"{dirpath}/")
    _walk(dirpath, "  ", 1)
    return "\n".join(result)


def select_model(client, required=True):
    """Let user pick a model from LM Studio.

    If required=True, exits on error. Otherwise returns None on error.
    Returns (model_key, context_length) or None.
    """
    try:
        models = client.list_models()
    except Exception as e:
        show_error(f"Cannot connect to LM Studio: {e}")
        if required:
            sys.exit(1)
        return None

    if not models:
        show_error("No models loaded in LM Studio.")
        if required:
            sys.exit(1)
        return None

    if len(models) == 1 and required:
        key, ctx = models[0]
        ctx_info = f", context: {ctx}" if ctx else ""
        show_info(f"Model: {key}{ctx_info}")
        return key, ctx

    console.print("\n[bold]Available models:[/]")
    for i, (key, ctx) in enumerate(models, 1):
        ctx_info = f" [dim](ctx: {ctx})[/]" if ctx else ""
        console.print(f"  [cyan]{i}[/]) {key}{ctx_info}")

    while True:
        try:
            choice = input(f"\nSelect model [1-{len(models)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            if required:
                sys.exit(0)
            return None
        if not choice:
            if not required:
                return None
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            key, ctx = models[int(choice) - 1]
            ctx_info = f", context: {ctx}" if ctx else ""
            show_info(f"Model: {key}{ctx_info}")
            return key, ctx
        console.print("[red]Invalid choice, try again.[/]")


def main():
    # --yolo / --dangerously-skip-permissions flag
    if "--yolo" in sys.argv or "--dangerously-skip-permissions" in sys.argv:
        executor.skip_permissions = True

    show_welcome()

    if executor.skip_permissions:
        show_yolo_enabled()

    global codebase_index, session_stats, added_files

    client = LMClient()
    model_key, model_ctx = select_model(client)
    client.model = model_key
    max_tokens = model_ctx if model_ctx else MAX_CONTEXT_TOKENS

    # Initialize codebase index and session stats
    codebase_index = CodebaseIndex()
    codebase_index.build(".")
    show_info(f"Codebase indexed: {codebase_index.file_count} files")
    session_stats = SessionStats()

    # Build system prompt with project memory
    project_memory = load_project_memory()
    system_prompt = SYSTEM_PROMPT.format(
            cwd=os.getcwd(),
            platform="Windows" if sys.platform == "win32" else "Linux/Mac",
            del_cmd="del" if sys.platform == "win32" else "rm",
        )
    if project_memory:
        system_prompt += "\n\n" + project_memory
        show_info("Loaded .openagent project memory.")
    messages = [{"role": "system", "content": system_prompt}]

    # Paste detection: track when buffer text last changed.
    # If Enter arrives < 50ms after a text change — it's a paste, insert newline.
    # If Enter arrives later — it's a real keypress, submit.
    _last_text_change = [0.0]

    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def _(event):
        buf = event.current_buffer
        text = buf.text
        now = time.monotonic()

        if not text.strip():
            buf.validate_and_handle()
        elif text.endswith("\\"):
            # Manual continuation: \ + Enter → newline
            buf.delete_before_cursor(1)
            buf.insert_text("\n")
        elif now - _last_text_change[0] < 0.05:
            # Rapid input (paste) — collect into buffer, don't submit
            buf.insert_text("\n")
        else:
            # Normal Enter — submit
            buf.validate_and_handle()

    @kb.add(Keys.Escape, Keys.Enter)
    def _(event):
        event.current_buffer.validate_and_handle()

    auto_suggest = NextActionSuggest()

    session = PromptSession(
        history=InMemoryHistory(),
        key_bindings=kb,
        multiline=True,
        lexer=CommandLexer(),
        completer=CommandCompleter(),
        complete_while_typing=False,
        auto_suggest=auto_suggest,
    )

    # Hook buffer text changes for paste detection
    def _on_text_changed(buf):
        _last_text_change[0] = time.monotonic()

    session.default_buffer.on_text_changed += _on_text_changed
    last_response = ""

    # User correction tracking for DPO
    correction_streak = 0     # consecutive user turns (reset on /clear)
    dpo_main_offered = False  # already offered DPO this streak
    dpo_main = None           # DPOCollector for main-loop corrections
    prev_turn_start = None    # message index where previous turn started

    while True:
        try:
            user_input = session.prompt("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            show_info("Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/exit":
            show_info("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            system_prompt = SYSTEM_PROMPT.format(
            cwd=os.getcwd(),
            platform="Windows" if sys.platform == "win32" else "Linux/Mac",
            del_cmd="del" if sys.platform == "win32" else "rm",
        )
            project_memory = load_project_memory()
            if project_memory:
                system_prompt += "\n\n" + project_memory
            messages = [{"role": "system", "content": system_prompt}]
            last_response = ""
            auto_suggest.clear()
            added_files = {}
            correction_streak = 0
            dpo_main_offered = False
            dpo_main = None
            prev_turn_start = None
            show_info("Conversation cleared.")
            continue
        elif user_input.lower() == "/help":
            show_help()
            continue
        elif user_input.lower() == "/save":
            save_code_from_response(last_response)
            continue
        elif user_input.lower() == "/undo":
            path, msg = undo_last()
            if path:
                show_info(msg)
            else:
                show_error(msg)
            continue
        elif user_input.lower().startswith("/add"):
            parts = user_input.split(None, 1)
            if len(parts) < 2:
                pinned_count = len(added_files)
                if pinned_count:
                    show_info(f"Pinned files ({pinned_count}): {', '.join(added_files.keys())}")
                else:
                    show_error("Usage: /add <file|dir|glob>  (e.g. /add *.py, /add src/)")
                continue
            target = parts[1].strip()

            # Case 1: directory
            if os.path.isdir(target):
                tree = _dir_tree(target)
                added_files[target] = tree
                messages.append({
                    "role": "user",
                    "content": f"[Pinned: {target}]\n{tree}",
                })
                show_info(f"Pinned directory: {target} ({len(added_files)} total)")
                continue

            # Case 2: single file
            if os.path.isfile(target):
                summary = _summarize_file(target)
                added_files[target] = summary
                messages.append({
                    "role": "user",
                    "content": f"[Pinned: {target}]\n{summary}",
                })
                show_info(f"Pinned file: {target} ({len(added_files)} total)")
                continue

            # Case 3: glob pattern
            import glob as glob_mod
            matches = glob_mod.glob(target, recursive=True)
            if not matches:
                show_error(f"No files matching: {target}")
                continue
            added_count = 0
            for fpath in sorted(matches):
                if os.path.isdir(fpath):
                    tree = _dir_tree(fpath)
                    added_files[fpath] = tree
                    messages.append({
                        "role": "user",
                        "content": f"[Pinned: {fpath}]\n{tree}",
                    })
                    added_count += 1
                elif os.path.isfile(fpath):
                    summary = _summarize_file(fpath)
                    added_files[fpath] = summary
                    messages.append({
                        "role": "user",
                        "content": f"[Pinned: {fpath}]\n{summary}",
                    })
                    added_count += 1
            if added_count:
                show_info(f"Pinned {added_count} item(s) ({len(added_files)} total)")
            continue
        elif user_input.lower() == "/model":
            result = select_model(client, required=False)
            if result:
                model_key, model_ctx = result
                client.model = model_key
                max_tokens = model_ctx if model_ctx else MAX_CONTEXT_TOKENS
            else:
                show_info("Model selection cancelled.")
            continue
        elif user_input.lower() == "/learn":
            show_info(show_learn_stats())
            continue
        elif user_input.lower() == "/cost":
            if session_stats:
                show_cost(session_stats.summary())
            else:
                show_info("No stats yet.")
            continue
        elif user_input.lower() == "/reindex":
            if codebase_index:
                codebase_index.build(".")
                show_info(f"Codebase re-indexed: {codebase_index.file_count} files")
            continue
        elif user_input.lower() == "/yolo":
            executor.skip_permissions = not executor.skip_permissions
            if executor.skip_permissions:
                show_yolo_enabled()
            else:
                show_yolo_disabled()
            continue
        elif user_input.lower().startswith("/save-session"):
            parts = user_input.split(None, 1)
            name = parts[1].strip() if len(parts) > 1 else datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join("data", "sessions")
            os.makedirs(session_dir, exist_ok=True)
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "model": client.model,
                "messages": messages,
            }
            session_path = os.path.join(session_dir, f"{name}.json")
            try:
                with open(session_path, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
                show_info(f"Session saved: {session_path}")
            except Exception as e:
                show_error(f"Failed to save session: {e}")
            continue
        elif user_input.lower().startswith("/load-session"):
            parts = user_input.split(None, 1)
            session_dir = os.path.join("data", "sessions")
            if len(parts) > 1:
                name = parts[1].strip()
                session_path = os.path.join(session_dir, f"{name}.json")
            else:
                # List available sessions
                if not os.path.isdir(session_dir):
                    show_error("No saved sessions found.")
                    continue
                files = sorted(f for f in os.listdir(session_dir) if f.endswith(".json"))
                if not files:
                    show_error("No saved sessions found.")
                    continue
                console.print("\n[bold]Saved sessions:[/]")
                for i, fname in enumerate(files, 1):
                    console.print(f"  [cyan]{i}[/]) {fname[:-5]}")
                try:
                    choice = input(f"\nLoad session [1-{len(files)}]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                if not choice.isdigit() or not (1 <= int(choice) <= len(files)):
                    show_error("Invalid choice.")
                    continue
                session_path = os.path.join(session_dir, files[int(choice) - 1])
            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                messages[:] = session_data["messages"]
                loaded_model = session_data.get("model", "")
                if loaded_model and loaded_model != client.model:
                    show_info(f"Session used model: {loaded_model} (current: {client.model})")
                show_info(f"Session loaded: {session_path} ({len(messages)} messages)")
            except FileNotFoundError:
                show_error(f"Session not found: {session_path}")
            except Exception as e:
                show_error(f"Failed to load session: {e}")
            continue
        elif user_input.startswith("/"):
            show_error(f"Unknown command: {user_input.split()[0]}")
            continue

        messages.append({"role": "user", "content": user_input})

        # Few-shot injection: find similar past successes and hint the model
        few_shot_block = build_few_shot_block(user_input)
        if few_shot_block:
            messages.append({"role": "system", "content": few_shot_block})

        # DPO anti-pattern injection: use past corrections immediately
        dpo_block = build_dpo_block(user_input)
        if dpo_block:
            messages.append({"role": "system", "content": dpo_block})

        # Auto-context: inject relevant files from codebase index
        auto_context_ids = []
        if codebase_index:
            relevant = codebase_index.query(user_input)
            for fpath in relevant:
                summary = _summarize_file(fpath)[:500]
                ctx_msg = {"role": "system", "content": f"[Auto-context: {fpath}]\n{summary}"}
                messages.append(ctx_msg)
                auto_context_ids.append(id(ctx_msg))

        # Auto-compact when approaching context limit
        if estimate_tokens(messages) > max_tokens * COMPACT_THRESHOLD:
            messages = compact_messages(client, messages)

        turn_start = len(messages)  # where model messages for this turn begin

        try:
            response_text = run_conversation_turn(client, messages, max_tokens=max_tokens)
            # Remove auto-context messages to avoid accumulation
            if auto_context_ids:
                messages[:] = [m for m in messages if id(m) not in auto_context_ids]
            if response_text:
                last_response = response_text
            # Show token usage
            show_token_usage(estimate_tokens(messages), max_tokens)
            # Generate suggestion in background (non-blocking)
            auto_suggest.clear()
            _msgs_snap = list(messages)
            _resp_snap = response_text or ""
            def _bg_suggest(msgs=_msgs_snap, resp=_resp_snap):
                try:
                    suggestion = _generate_suggestion(client, msgs, resp)
                    auto_suggest.set(suggestion)
                    try:
                        session.app.invalidate()
                    except Exception:
                        pass
                except Exception:
                    pass
            threading.Thread(target=_bg_suggest, daemon=True).start()
        except (KeyboardInterrupt, EscInterrupt):
            show_info("\nTask stopped.")
            auto_suggest.clear()
            continue
        except Exception as e:
            show_error(f"API error: {e}")
            # Remove the user message that caused the error
            messages.pop()
            auto_suggest.clear()
            continue

        # --- User correction tracking for DPO ---
        correction_streak += 1

        # Offer DPO at 3+ consecutive corrections
        if correction_streak >= 3 and not dpo_main_offered:
            if show_dpo_prompt("Вы исправляете модель 3+ раз подряд."):
                user_task = _find_user_task(messages)
                dpo_main = DPOCollector(user_task, messages[0]["content"])
                dpo_main.activate()
                # Inject course correction into current conversation
                messages.append({
                    "role": "system",
                    "content": (
                        "The user has corrected you a few times. Adjust your approach:\n"
                        "1) Use list_files to check what exists.\n"
                        "2) Use read_file to read files before editing.\n"
                        "3) Make decisions yourself — do not ask the user.\n"
                        "4) Use the <function=tool_name> format for ALL actions.\n"
                        "5) Do NOT write shell commands as text.\n"
                        "Try a different strategy this time."
                    ),
                })
            dpo_main_offered = True

        # Create DPO pair: previous turn (rejected) → current turn (chosen)
        if dpo_main and dpo_main.active and prev_turn_start is not None:
            prev_slice = messages[prev_turn_start:turn_start]
            curr_slice = messages[turn_start:]
            if prev_slice and curr_slice:
                dpo_main.add_rejected(prev_slice)
                dpo_main.set_chosen(curr_slice)
                saved = dpo_main.save()
                if saved:
                    total = DPOCollector.count_pairs()
                    show_dpo_saved(saved, total)

        prev_turn_start = turn_start


if __name__ == "__main__":
    main()
