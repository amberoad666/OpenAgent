"""Codebase RAG — lightweight TF-IDF index for auto-context injection.

No external dependencies. Builds an in-memory inverted index of all text files
in the project, then returns the top-K most relevant files for a given query.
"""

import math
import os
import re

# Directories and extensions to skip
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode", ".tox", "dist", "build", "egg-info"}
_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
    ".pyc", ".pyo", ".class", ".whl",
    ".db", ".sqlite", ".sqlite3",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".woff", ".woff2", ".ttf", ".eot",
}
_MAX_FILE_SIZE = 100 * 1024  # 100 KB

_TOKEN_RE = re.compile(r'[a-zA-Z_]\w{1,}')


def _tokenize(text):
    """Extract identifier-like tokens from text (lowercased, len >= 2)."""
    return _TOKEN_RE.findall(text.lower())


class CodebaseIndex:
    """TF-IDF index over project files for auto-context retrieval."""

    def __init__(self):
        self._docs = {}       # filepath → list of tokens
        self._idf = {}        # token → IDF value
        self._doc_count = 0
        self._stale = False

    def build(self, root="."):
        """Scan files under *root* and build the inverted index."""
        self._docs.clear()
        self._idf.clear()

        root = os.path.abspath(root)
        doc_freq = {}  # token → number of docs containing it

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip unwanted directories (in-place filter)
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _BINARY_EXTS:
                    continue

                fpath = os.path.join(dirpath, fname)

                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue
                if size > _MAX_FILE_SIZE or size == 0:
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except (OSError, UnicodeDecodeError):
                    continue

                tokens = _tokenize(text)
                if not tokens:
                    continue

                rel = os.path.relpath(fpath, root)
                self._docs[rel] = tokens

                unique = set(tokens)
                for tok in unique:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self._doc_count = len(self._docs)
        if self._doc_count == 0:
            return

        # Compute IDF = log(N / df)
        for tok, df in doc_freq.items():
            self._idf[tok] = math.log(self._doc_count / df)

        self._stale = False

    def query(self, text, top_k=3):
        """Return top-K file paths most relevant to *text*.

        Uses TF * IDF² scoring — rare tokens (function names) weigh more.
        """
        if self._stale:
            self.rebuild()

        if not self._docs:
            return []

        q_tokens = _tokenize(text)
        if not q_tokens:
            return []

        # Query token set for fast lookup
        q_set = set(q_tokens)

        scores = {}
        for filepath, doc_tokens in self._docs.items():
            score = 0.0
            # Build term frequency for this doc (only for query tokens)
            tf = {}
            for tok in doc_tokens:
                if tok in q_set:
                    tf[tok] = tf.get(tok, 0) + 1

            if not tf:
                continue

            doc_len = len(doc_tokens)
            for tok, count in tf.items():
                idf = self._idf.get(tok, 0.0)
                # TF normalized by doc length, IDF squared
                score += (count / doc_len) * (idf * idf)

            if score > 0:
                scores[filepath] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [path for path, _ in ranked[:top_k]]

    def mark_stale(self):
        """Mark the index as stale — will rebuild on next query()."""
        self._stale = True

    def rebuild(self):
        """Rebuild the index from the same root (current directory)."""
        self.build(".")

    @property
    def file_count(self):
        return self._doc_count
