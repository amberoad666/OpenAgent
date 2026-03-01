import difflib
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# On Windows, Rich uses LegacyWindowsTerm which fails on characters
# outside the console codepage (e.g. ₽ U+20BD on cp1251).
# force_terminal=True makes Rich use ANSI escape codes instead,
# which modern Windows 10+ terminals support natively.
if sys.platform == "win32":
    # Enable UTF-8 output if possible
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # Enable VT100 processing on Windows
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

console = Console(force_terminal=True)


def _read_key():
    """Read a single keypress without waiting for Enter.

    Returns the character, or '\\x1b' for ESC.
    """
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        # Handle special keys (arrows etc.) — read second byte and ignore
        if ch in ('\x00', '\xe0'):
            msvcrt.getwch()
            return ""
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


def show_welcome():
    """Display welcome message."""
    console.print(
        Panel(
            "[bold cyan]OpenAgent[/] — AI coding assistant (local LLM)\n"
            "Type your message to chat. Commands: /help, /save, /save-session, /load-session, /clear, /yolo, /undo, /add, /model, /learn, /cost, /reindex, /exit",
            title="Welcome",
            border_style="cyan",
        )
    )


def show_help():
    """Display help information."""
    console.print(
        Panel(
            "[bold]Commands:[/]\n"
            "  [bold]/save[/]       — Save code from last response to a file\n"
            "  [bold]/save-session[/] — Save current session to file\n"
            "  [bold]/load-session[/] — Load a saved session\n"
            "  [bold]/clear[/]      — Clear conversation history\n"
            "  [bold]/undo[/]       — Undo last file change\n"
            "  [bold]/add <path>[/] — Pin file(s) to context (persists through compaction)\n"
            "  [bold]/add[/]          — List pinned files\n"
            "  [bold]/model[/]      — Switch to a different model\n"
            "  [bold]/learn[/]      — Show learning statistics (few-shot + DPO)\n"
            "  [bold]/cost[/]       — Show session token usage and stats\n"
            "  [bold]/reindex[/]    — Rebuild codebase index for auto-context\n"
            "  [bold]/yolo[/]       — Toggle skip-permissions mode\n"
            "  [bold]/help[/]       — Show this help message\n"
            "  [bold]/exit[/]       — Exit the program\n\n"
            "[bold]Input:[/]\n"
            "  [bold]Enter[/]       — Send message (single line)\n"
            "  [bold]\\\\+Enter[/]   — Add new line (type \\\\ at end of line)\n"
            "  [bold]Esc+Enter[/]  — Send multiline message\n"
            "  [bold]Right Arrow[/] — Accept suggested next action\n\n"
            "The assistant can read/write files, run commands, search code, and search the web.\n"
            "You will be asked to confirm dangerous operations.",
            title="Help",
            border_style="blue",
        )
    )


def show_assistant_message(text):
    """Render assistant's response as Markdown."""
    if text and text.strip():
        console.print()
        console.print(Markdown(text))
        console.print()


def show_streaming_text(text):
    """Print text during streaming (raw, no newline)."""
    console.print(text, end="", highlight=False)


def show_tool_call(name, arguments):
    """Display a tool call in a styled panel."""
    args_display = ""
    for key, value in arguments.items():
        val_str = str(value)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        args_display += f"  [bold]{key}[/]: {val_str}\n"

    console.print()
    console.print(
        Panel(
            args_display.rstrip(),
            title=f"Tool: {name}",
            border_style="yellow",
        )
    )


def show_running_hint():
    """Show hint on how to interrupt a running command."""
    console.print("[dim]  Ctrl+C — остановить[/]")


def show_tool_result(name, result):
    """Display tool execution result."""
    display = result
    if len(display) > 1000:
        display = display[:1000] + f"\n... ({len(result)} chars total)"
    console.print(
        Panel(
            display,
            title=f"Result: {name}",
            border_style="green",
        )
    )


def confirm_action(name, arguments):
    """Ask user to confirm a dangerous action.

    Single-keypress input: y=approve, n/Esc=skip, e=explain.
    No Enter needed for y/n/Esc.

    Returns (approved, feedback):
      - (True, None)   — user approved
      - (False, str)   — user declined with a reason/instruction
    """
    console.print(
        Text(f"  Allow {name}? ", style="bold yellow"),
        end="",
    )
    console.print(Text("[y/n/e/Esc]", style="dim"))

    try:
        key = _read_key()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False, "Action cancelled by user."

    if key.lower() == "y":
        console.print(Text("  y", style="bold green"))
        return True, None

    if key.lower() == "e":
        console.print(Text("  e", style="bold cyan"))
        console.print(Text("  What should be changed? ", style="bold yellow"), end="")
        try:
            feedback = input().strip()
        except (EOFError, KeyboardInterrupt):
            return False, "Action cancelled by user."
        if feedback:
            return False, f"User rejected this action. User's instruction: {feedback}"
        return False, "User rejected this action without explanation."

    # n, Esc, or anything else → decline
    if key == "\x1b":
        console.print(Text("  Esc", style="bold red"))
    else:
        console.print(Text("  n", style="bold red"))

    console.print(
        Panel(
            "[bold]1[/]) Skip — just skip this action\n"
            "[bold]2[/]) Explain — tell the agent what's wrong",
            title="Action declined",
            border_style="red",
        )
    )

    try:
        key2 = _read_key()
    except (EOFError, KeyboardInterrupt):
        return False, "Action cancelled by user."

    if key2 == "2":
        console.print(Text("  2", style="bold cyan"))
        console.print(Text("  What's wrong? ", style="bold yellow"), end="")
        try:
            feedback = input().strip()
        except (EOFError, KeyboardInterrupt):
            return False, "Action cancelled by user."
        if feedback:
            return False, f"User rejected this action. User's feedback: {feedback}"

    if key2 == "\x1b":
        console.print(Text("  Esc", style="bold red"))
    else:
        console.print(Text("  1", style="bold yellow"))

    return False, "Action cancelled by user."


def show_yolo_enabled():
    """Display warning that skip-permissions mode is ON."""
    console.print(
        Panel(
            "[bold]Skip-permissions mode enabled.[/]\n"
            "All tools will execute without confirmation.\n"
            "Type [bold]/yolo[/] again to disable.",
            title="YOLO MODE",
            border_style="red",
        )
    )


def show_yolo_disabled():
    """Display confirmation that skip-permissions mode is OFF."""
    console.print(
        Panel(
            "Confirmations restored. Dangerous tools will ask before executing.",
            title="YOLO mode disabled",
            border_style="green",
        )
    )


def show_token_usage(used, total):
    """Display token usage bar: green <50%, yellow 50-75%, red >75%."""
    if total <= 0:
        return
    pct = used / total
    if used >= 1000:
        used_str = f"{used / 1000:.1f}k"
    else:
        used_str = str(used)
    if total >= 1000:
        total_str = f"{total / 1000:.1f}k"
    else:
        total_str = str(total)
    pct_str = f"{pct * 100:.0f}%"
    if pct < 0.5:
        color = "green"
    elif pct < 0.75:
        color = "yellow"
    else:
        color = "red"
    console.print(f"[dim]tokens:[/] [{color}]{used_str} / {total_str} ({pct_str})[/{color}]")


def show_error(message):
    """Display an error message."""
    console.print(f"[bold red]Error:[/] {message}")


def show_info(message):
    """Display an info message."""
    console.print(f"[dim]{message}[/]")


def show_dpo_prompt(reason=None):
    """Ask user if they want to enable DPO learning mode.

    Returns True if user agrees. Single-keypress: y=yes, anything else=no.
    """
    msg = reason or "Модель не справляется (3+ nudge)."
    console.print(
        Panel(
            f"[bold]{msg}[/]\n"
            "Включить режим обучения? Плохие и хорошие ответы\n"
            "сохранятся и будут использоваться как антипаттерны.",
            title="Learning mode",
            border_style="magenta",
        )
    )
    console.print(Text("  Включить? [y/N] ", style="bold magenta"), end="")
    try:
        key = _read_key()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    if key.lower() in ("y", "д"):
        console.print(Text("y", style="bold green"))
        return True
    console.print(Text("n", style="bold red"))
    return False


def show_dpo_saved(count, total):
    """Display DPO save confirmation."""
    console.print(f"[magenta]DPO: +{count} пар (всего: {total})[/magenta]")


def show_diff(old_text, new_text, filepath):
    """Display a unified diff of file changes."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{filepath}", tofile=f"b/{filepath}")
    diff_text = "".join(diff)
    if not diff_text:
        return
    try:
        syn = Syntax(diff_text, "diff", theme="monokai")
        console.print(Panel(syn, title=f"Diff: {filepath}", border_style="magenta"))
    except UnicodeEncodeError:
        # cp1251 can't encode some chars (₽, emoji, etc.) — replace them
        safe = diff_text.encode("cp1251", errors="replace").decode("cp1251")
        console.print(Panel(safe, title=f"Diff: {filepath}", border_style="magenta"))
    except Exception:
        try:
            console.print(Panel(diff_text, title=f"Diff: {filepath}", border_style="magenta"))
        except UnicodeEncodeError:
            safe = diff_text.encode("cp1251", errors="replace").decode("cp1251")
            console.print(Panel(safe, title=f"Diff: {filepath}", border_style="magenta"))


def show_cost(stats):
    """Display session token usage and performance stats."""
    def _fmt_tokens(n):
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    lines = []
    lines.append(f"  Requests:          {stats['requests']}")
    lines.append(f"  Prompt tokens:     ~{_fmt_tokens(stats['prompt_tokens'])}")
    lines.append(f"  Completion tokens: ~{_fmt_tokens(stats['completion_tokens'])}")
    lines.append(f"  Total tokens:      ~{_fmt_tokens(stats['total_tokens'])}")
    if stats.get("avg_speed"):
        lines.append(f"  Avg speed:         {stats['avg_speed']:.1f} tok/s")
    if stats.get("avg_time"):
        lines.append(f"  Avg gen time:      {stats['avg_time']:.1f}s")
    if stats.get("session_time"):
        mins = int(stats["session_time"] // 60)
        secs = int(stats["session_time"] % 60)
        lines.append(f"  Session time:      {mins}m {secs}s")

    console.print(
        Panel(
            "\n".join(lines),
            title="Session Stats",
            border_style="blue",
        )
    )
