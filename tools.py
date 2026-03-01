import os
import glob
import subprocess
import re
import sys
import threading

# On Windows, CMD outputs text in OEM codepage (e.g. cp866 for Russian),
# but Python's text=True uses the ANSI codepage (e.g. cp1251). Detect both.
if sys.platform == "win32":
    import ctypes
    _OEM_ENCODING = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    _ANSI_ENCODING = f"cp{ctypes.windll.kernel32.GetACP()}"
else:
    _OEM_ENCODING = None
    _ANSI_ENCODING = None


def _decode_output(data):
    """Decode subprocess output bytes on Windows, handling OEM/ANSI ambiguity.

    Tries OEM codepage first (most commands), falls back to ANSI,
    then UTF-8 as last resort.
    """
    if not data:
        return ""
    if _OEM_ENCODING is None:
        return data.decode("utf-8", errors="replace")
    # Try OEM (cp866) — correct for most commands and error messages
    try:
        text = data.decode(_OEM_ENCODING)
        # Heuristic: if the result has box-drawing chars mixed with Cyrillic,
        # it's probably ANSI data decoded as OEM — retry with ANSI
        if _has_box_drawing(text) and _has_cyrillic(text):
            return data.decode(_ANSI_ENCODING, errors="replace")
        return text
    except (UnicodeDecodeError, LookupError):
        pass
    # Try ANSI (cp1251)
    try:
        return data.decode(_ANSI_ENCODING, errors="replace")
    except (UnicodeDecodeError, LookupError):
        pass
    return data.decode("utf-8", errors="replace")


def _has_box_drawing(text):
    """Check if text contains Unicode box-drawing characters (U+2500-U+257F, U+2580-U+259F)."""
    return any('\u2500' <= ch <= '\u259f' for ch in text[:100])


def _has_cyrillic(text):
    """Check if text contains Cyrillic characters."""
    return any('\u0400' <= ch <= '\u04ff' for ch in text[:100])

# --- File backups for /undo ---
_file_backups = {}  # path → previous content (or None if file didn't exist)

# --- Last edit diff (for UI display) ---
_last_edit_diff = None  # (path, old_content, new_content) or None

def get_last_edit_diff():
    """Return and clear the last edit diff. Returns (path, old, new) or None."""
    global _last_edit_diff
    result = _last_edit_diff
    _last_edit_diff = None
    return result

# --- File read cache (cleared each conversation turn) ---
_read_cache = {}  # abs_path → content


def _invalidate_cache(path):
    """Remove a path from the read cache."""
    abs_path = os.path.abspath(path)
    _read_cache.pop(abs_path, None)


def clear_read_cache():
    """Clear the entire read cache. Called at the start of each conversation turn."""
    _read_cache.clear()


def undo_last():
    """Restore the last modified file to its previous state.

    Returns a message describing what was undone, or an error.
    """
    if not _file_backups:
        return None, "Nothing to undo."
    path, old_content = _file_backups.popitem()
    try:
        if old_content is None:
            # File didn't exist before — delete it
            if os.path.exists(path):
                os.remove(path)
                return path, f"Deleted {path} (was newly created)."
            return path, f"{path} already gone."
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(old_content)
            return path, f"Restored {path} to previous version."
    except Exception as e:
        return path, f"Undo failed: {e}"


def _backup_file(path):
    """Save current file content before modification."""
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                _file_backups[abs_path] = f.read()
        except Exception:
            _file_backups[abs_path] = None
    else:
        _file_backups[abs_path] = None


# --- Tool definitions (OpenAI function calling format) ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing old_text with new_text. The old_text must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to edit."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The text to replace old_text with."
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash/shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory matching an optional glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The directory to list files in. Defaults to current directory.",
                        "default": "."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py', '**/*.js'). Defaults to '*'.",
                        "default": "*"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a regex pattern in files within a directory (like grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory or file to search in. Defaults to current directory.",
                        "default": "."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns top 5 results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
]

# --- Tool implementations ---


def _safe_path(path):
    """Sanitize path: strip leading slashes to avoid writing to root."""
    path = os.path.expanduser(path)
    # Convert absolute unix-style paths like "/file.txt" to relative
    if path.startswith("/") and not os.path.exists(path):
        path = path.lstrip("/")
    return path


def read_file(path):
    """Read file contents."""
    path = _safe_path(path)
    abs_path = os.path.abspath(path)
    if abs_path in _read_cache:
        return _read_cache[abs_path]
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if not os.path.isfile(path):
        return f"Error: Not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = content.split("\n")
        if len(lines) > 2000:
            result = f"[File has {len(lines)} lines, showing first 2000]\n" + "\n".join(lines[:2000])
            _read_cache[abs_path] = result
            return result
        _read_cache[abs_path] = content
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path, content):
    """Write content to a file."""
    path = _safe_path(path)
    _invalidate_cache(path)
    _backup_file(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _find_indent_fuzzy(content, old_text):
    """Find a region in content matching old_text, ignoring indentation.

    Compares stripped lines to tolerate spaces/tabs differences.
    Returns the matching substring from content (with original indentation),
    or None if no match found.
    """
    old_lines = [l.strip() for l in old_text.strip().splitlines() if l.strip()]
    if not old_lines:
        return None

    content_lines = content.splitlines(True)  # keep line endings

    for i in range(len(content_lines)):
        if content_lines[i].strip() == old_lines[0]:
            # Try matching consecutive non-blank lines from here
            matched = []
            j = 0   # index in old_lines
            k = i   # index in content_lines
            while j < len(old_lines) and k < len(content_lines):
                stripped = content_lines[k].strip()
                if not stripped:
                    # Skip blank lines in content
                    matched.append(content_lines[k])
                    k += 1
                    continue
                if stripped == old_lines[j]:
                    matched.append(content_lines[k])
                    j += 1
                    k += 1
                else:
                    break

            if j == len(old_lines):
                # Full match — return original text from content
                result = ''.join(matched)
                if result.endswith('\n'):
                    result = result[:-1]
                return result

    return None


def edit_file(path, old_text, new_text):
    """Replace old_text with new_text in a file."""
    global _last_edit_diff
    _last_edit_diff = None
    path = _safe_path(path)
    _invalidate_cache(path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}. Use write_file to create new files."
    _backup_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        content_before_edit = content

        # If file is empty or whitespace-only, just write new_text
        if not content.strip():
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
            return f"File {path} was empty — wrote new content."

        count = 1
        if old_text in content:
            # Exact match
            count = content.count(old_text)
            content = content.replace(old_text, new_text, 1)
        elif old_text.strip() and old_text.strip() in content:
            # Stripped match (whitespace difference)
            count = content.count(old_text.strip())
            content = content.replace(old_text.strip(), new_text, 1)
        else:
            # Indent-fuzzy match: same lines, different indentation
            fuzzy = _find_indent_fuzzy(content, old_text)
            if fuzzy:
                content = content.replace(fuzzy, new_text, 1)
            else:
                # Try to find the closest matching line to help the model
                hint_lines = []
                for line in old_text.strip().splitlines():
                    line_s = line.strip()
                    if line_s and line_s in content:
                        # Find the actual full line in the file
                        for fl in content.splitlines():
                            if line_s in fl:
                                hint_lines.append(fl)
                                break
                if hint_lines:
                    found = "\n".join(hint_lines[:5])
                    return (
                        f"Error: old_text not found in {path} (whitespace or context mismatch). "
                        f"Similar lines found in file:\n{found}\n"
                        f"Use read_file(\"{path}\") first, then copy the EXACT text for old_text."
                    )
                snippet = content[:300].replace('\n', '\\n')
                return (
                    f"Error: old_text not found in {path}. "
                    f"File starts with: {snippet!r}\n"
                    f"Use read_file(\"{path}\") to see exact content, then edit_file with correct old_text."
                )

        _last_edit_diff = (path, content_before_edit, content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        msg = f"Successfully edited {path}"
        if count > 1:
            msg += f" (replaced first of {count} occurrences)"
        return msg
    except Exception as e:
        return f"Error editing file: {e}"


def _check_pip_install(command):
    """If command is 'pip install ...', check which packages are already installed.

    Returns (modified_command, message):
      - If all installed: (None, "Already installed: x, y")
      - If some missing: (pip install only_missing, "Already installed: x. Installing: y")
      - If not a pip install: (command, None)
    """
    # Match: pip install pkg1 pkg2 ... (ignore flags like --upgrade)
    m = re.match(r'^pip\s+install\s+(.+)$', command.strip())
    if not m:
        return command, None

    args = m.group(1).split()
    packages = [a for a in args if not a.startswith('-')]
    flags = [a for a in args if a.startswith('-')]

    if not packages:
        return command, None

    installed = []
    missing = []
    for pkg in packages:
        # pip show returns 0 if installed, 1 if not
        r = subprocess.run(
            f"pip show {pkg}",
            shell=True, capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            installed.append(pkg)
        else:
            missing.append(pkg)

    parts = []
    if installed:
        parts.append(f"Already installed: {', '.join(installed)}")
    if missing:
        parts.append(f"Installing: {', '.join(missing)}")
        new_cmd = "pip install " + " ".join(flags + missing)
        return new_cmd, "\n".join(parts)
    else:
        return None, "\n".join(parts)


# Commands that are expected to run indefinitely (servers, bots, watchers).
# If the process is still alive after LONG_RUNNING_WAIT, we report success.
LONG_RUNNING_WAIT = 5  # seconds to wait before declaring "started OK"

_LONG_RUNNING_RE = re.compile(
    r'python[3]?\s+\S+\.py'   # python script.py (bots, servers)
    r'|node\s+\S+\.js'        # node app.js
    r'|npm\s+(?:start|run)'   # npm start / npm run dev
    r'|uvicorn\s'             # uvicorn
    r'|gunicorn\s'            # gunicorn
    r'|flask\s+run'           # flask run
    r'|streamlit\s+run'       # streamlit run
    r'|gradio\s'              # gradio
    r'|java\s+-jar'           # java -jar
    r'|docker\s+(?:run|compose\s+up)',  # docker run/compose up
    re.IGNORECASE,
)

# Streaming bash state
_last_bash_was_streamed = False


def was_last_bash_streamed():
    """Return True if the last run_bash call streamed output to stdout."""
    return _last_bash_was_streamed


def run_bash(command):
    """Execute a bash command. Auto-checks pip install to skip already installed packages.

    For potentially long-running commands (bots, servers), uses Popen with a
    short wait: if the process is still alive after a few seconds, returns
    partial output and reports it's running (instead of blocking for 30s).
    """
    # Smart pip install: check before installing
    command, pip_msg = _check_pip_install(command)
    if command is None:
        # All packages already installed
        return pip_msg

    is_long_running = bool(_LONG_RUNNING_RE.search(command))

    # Longer timeout for pip install
    timeout = 120 if command.strip().startswith("pip ") else 30

    if is_long_running:
        return _run_long_lived(command, pip_msg)

    global _last_bash_was_streamed
    _last_bash_was_streamed = False

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Kill process after timeout
        timer = threading.Timer(timeout, proc.kill)
        timer.start()

        output_chunks = []
        line_count = 0
        try:
            for raw_line in iter(proc.stdout.readline, b''):
                decoded = _decode_output(raw_line)
                output_chunks.append(decoded)
                line_count += 1
                sys.stdout.write("  " + decoded)
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            timer.cancel()

        proc.wait()

        if line_count > 0:
            _last_bash_was_streamed = True

        output = ""
        if pip_msg:
            output += pip_msg + "\n"
        stdout_text = "".join(output_chunks)
        if stdout_text:
            output += stdout_text
        if proc.returncode != 0:
            output += f"\n[Exit code: {proc.returncode}]"
        return output if output else "[No output]"
    except Exception as e:
        return f"Error running command: {e}"


def _run_long_lived(command, pip_msg=None):
    """Run a potentially long-lived process (bot, server, etc.).

    Starts the process, waits LONG_RUNNING_WAIT seconds:
    - If it exits quickly (crash/error) → return full output.
    - If still running → return partial output + PID, leave it running.
    """
    import time
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        return f"Error starting process: {e}"

    # Wait a few seconds to see if it crashes immediately
    try:
        proc.wait(timeout=LONG_RUNNING_WAIT)
    except subprocess.TimeoutExpired:
        # Still running — this is expected for bots/servers
        # Read whatever output is available without blocking
        partial_out = b""
        partial_err = b""
        import selectors
        try:
            # Non-blocking read of available output
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            sel.register(proc.stderr, selectors.EVENT_READ)
            while True:
                ready = sel.select(timeout=0.1)
                if not ready:
                    break
                for key, _ in ready:
                    data = key.fileobj.read1(4096) if hasattr(key.fileobj, 'read1') else b""
                    if key.fileobj == proc.stdout:
                        partial_out += data
                    else:
                        partial_err += data
            sel.close()
        except Exception:
            pass

        output = ""
        if pip_msg:
            output += pip_msg + "\n"
        out_text = _decode_output(partial_out)
        err_text = _decode_output(partial_err)
        if out_text:
            output += out_text
        if err_text:
            output += ("" if not output else "\n") + err_text
        pid = proc.pid
        if sys.platform == "win32":
            kill_cmd = f"taskkill /PID {pid} /F"
        else:
            kill_cmd = f"kill {pid}"
        output += f"\n[Process running, PID: {pid}]"
        output += f"\n[To stop: run_bash(command=\"{kill_cmd}\")]"
        return output if output else f"[Process started, PID: {pid}]\n[To stop: run_bash(command=\"{kill_cmd}\")]"

    # Process exited within LONG_RUNNING_WAIT — probably an error
    stdout = _decode_output(proc.stdout.read())
    stderr = _decode_output(proc.stderr.read())
    output = ""
    if pip_msg:
        output += pip_msg + "\n"
    if stdout:
        output += stdout
    if stderr:
        output += ("" if not output else "\n") + stderr
    if proc.returncode != 0:
        output += f"\n[Exit code: {proc.returncode}]"
    return output if output else "[No output]"


def list_files(directory=".", pattern="*"):
    """List files matching a glob pattern."""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return f"Error: Directory not found: {directory}"
    try:
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern, recursive=True)
        files.sort()
        if not files:
            return f"No files matching '{pattern}' in {directory}"
        if len(files) > 200:
            return "\n".join(files[:200]) + f"\n... and {len(files) - 200} more files"
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"


def web_search(query):
    """Search the web using DuckDuckGo. Returns top 5 results."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        # Auto-install on first use
        subprocess.run(
            "pip install duckduckgo_search",
            shell=True, capture_output=True, text=True, timeout=60,
        )
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Error: Failed to install duckduckgo_search. Run: pip install duckduckgo_search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   {r.get('href', '')}")
            lines.append(f"   {r.get('body', '')}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Error searching web: {e}"


def search_files(pattern, path="."):
    """Search for a regex pattern in files."""
    path = os.path.expanduser(path)
    results = []
    max_results = 100

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    try:
        if os.path.isfile(path):
            files_to_search = [path]
        elif os.path.isdir(path):
            files_to_search = []
            for root, dirs, filenames in os.walk(path):
                # Skip hidden directories and common non-text dirs
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")]
                for fname in filenames:
                    files_to_search.append(os.path.join(root, fname))
        else:
            return f"Error: Path not found: {path}"

        for fpath in files_to_search:
            if len(results) >= max_results:
                break
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if compiled.search(line):
                            results.append(f"{fpath}:{i}: {line.rstrip()}")
                            if len(results) >= max_results:
                                break
            except (OSError, UnicodeDecodeError):
                continue

        if not results:
            return f"No matches found for '{pattern}' in {path}"
        output = "\n".join(results)
        if len(results) == max_results:
            output += f"\n... (limited to {max_results} results)"
        return output
    except Exception as e:
        return f"Error searching files: {e}"
