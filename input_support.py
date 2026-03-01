"""Command highlighting, Tab completion, and auto-suggest for the prompt."""

import os
import glob as glob_mod

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.document import Document

# Single source of truth for all slash commands
COMMANDS = ["/help", "/save", "/save-session", "/load-session", "/clear", "/yolo", "/undo", "/add", "/model", "/learn", "/cost", "/reindex", "/exit"]
COMMANDS_SET = set(COMMANDS)


class NextActionSuggest(AutoSuggest):
    """Shows a ghost-text suggestion for the next user action.

    The suggestion is only visible when the input field is empty.
    Press Right Arrow to accept it.
    """

    def __init__(self):
        self.suggestion_text = ""

    def get_suggestion(self, buffer, document):
        if not document.text and self.suggestion_text:
            return Suggestion(self.suggestion_text)
        return None

    def set(self, text):
        self.suggestion_text = text.strip()

    def clear(self):
        self.suggestion_text = ""


class CommandLexer(Lexer):
    """Highlights /commands on the input line.

    Known commands → cyan bold, unknown /word → red bold.
    """

    def lex_document(self, document):
        def get_line(lineno):
            line = document.lines[lineno]
            # Only highlight the first line (where commands are typed)
            if lineno != 0:
                return [("", line)]
            stripped = line.strip()
            if stripped.startswith("/"):
                word = stripped.split()[0] if stripped.split() else stripped
                if word in COMMANDS_SET:
                    return [("bold fg:ansibrightcyan", line)]
                else:
                    return [("bold fg:ansired", line)]
            return [("", line)]

        return get_line


class CommandCompleter(Completer):
    """Tab completion for /commands and file paths after /add."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        # /add <path> — complete file paths
        if text.startswith("/add "):
            partial = text[5:]  # after "/add "
            # Expand glob-like or partial paths
            if partial:
                candidates = glob_mod.glob(partial + "*")
            else:
                candidates = os.listdir(".")
            for path in sorted(candidates)[:20]:
                display = path
                if os.path.isdir(path):
                    display += "/"
                yield Completion(path, start_position=-len(partial), display=display)
            return

        # Regular command completion
        word = text.split()[0] if text.split() else text
        if " " not in text:
            for cmd in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
