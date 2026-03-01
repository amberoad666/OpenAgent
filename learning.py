"""Few-shot memory and DPO data collection for OpenAgent self-learning."""

import json
import os
import re
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FEW_SHOT_FILE = os.path.join(DATA_DIR, "few_shot.jsonl")
DPO_FILE = os.path.join(DATA_DIR, "dpo_pairs.jsonl")

MAX_FEW_SHOT_EXAMPLES = 200  # max entries in few_shot.jsonl before rotation

# Russian + English stop words for keyword extraction
_STOP_WORDS = {
    # Russian
    "и", "в", "на", "с", "по", "для", "из", "к", "у", "о", "от", "за", "не",
    "что", "как", "это", "но", "все", "так", "его", "мне", "мой", "ты", "он",
    "она", "они", "мы", "вы", "бы", "уже", "да", "нет", "если", "же", "ещё",
    "тоже", "только", "вот", "при", "до", "после", "нужно", "надо", "можно",
    "будет", "была", "был", "были", "есть", "быть", "этот", "эта", "эти",
    "тот", "та", "те", "свой", "свою", "себя", "чтобы", "потом", "когда",
    "где", "там", "тут", "сюда", "очень", "просто", "пожалуйста", "сделай",
    "создай", "напиши", "покажи", "помоги",
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "and", "or", "but",
    "if", "then", "else", "when", "up", "out", "no", "not", "so", "just",
    "also", "how", "all", "each", "every", "both", "few", "more", "most",
    "some", "any", "please", "make", "create", "write", "show", "help",
}


def _ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _extract_keywords(text):
    """Extract meaningful keywords from text, filtering stop words."""
    # Normalize: lowercase, split on non-alphanumeric (keeping cyrillic)
    words = re.findall(r'[a-zA-Zа-яА-ЯёЁ0-9_\-.]+', text.lower())
    # Filter stop words and very short words
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _keyword_overlap(a, b):
    """Count number of common keywords between two keyword lists."""
    return len(set(a) & set(b))


# ── Tier 1: Few-shot memory ──────────────────────────────────────────────

def save_few_shot(task, tool_chain, response, rounds):
    """Save a successful task execution as a few-shot example.

    Only saves if there were actual tool calls and it completed quickly.
    """
    if not task or not tool_chain:
        return
    if rounds > 5:
        return  # too many rounds = not a clean example

    _ensure_data_dir()

    keywords = _extract_keywords(task)
    if not keywords:
        return

    # Build compact trace
    steps = []
    for tc in tool_chain:
        tool = tc.get("tool", "?")
        args = tc.get("args", {})
        # Compact args representation
        arg_str = ", ".join(f"{v}" for v in args.values())[:60]
        mark = "OK" if tc.get("success") else "FAIL"
        steps.append(f"{tool}({arg_str}) {mark}")

    compact_trace = f"user: {task[:80]} -> " + " -> ".join(steps) + " -> done"

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task[:200],
        "keywords": keywords[:10],
        "tool_chain": tool_chain,
        "rounds": rounds,
        "compact_trace": compact_trace,
    }

    # Append to file
    try:
        with open(FEW_SHOT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return

    # Rotate if too many entries: keep last MAX entries
    _rotate_few_shot()


def _rotate_few_shot():
    """Keep only the last MAX_FEW_SHOT_EXAMPLES entries."""
    try:
        with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_FEW_SHOT_EXAMPLES:
            with open(FEW_SHOT_FILE, "w", encoding="utf-8") as f:
                f.writelines(lines[-MAX_FEW_SHOT_EXAMPLES:])
    except Exception:
        pass


def find_similar_examples(task, max_results=2):
    """Find few-shot examples similar to the given task.

    Returns list of compact_trace strings, sorted by relevance.
    """
    if not os.path.isfile(FEW_SHOT_FILE):
        return []

    task_keywords = _extract_keywords(task)
    if not task_keywords:
        return []

    scored = []
    try:
        with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_keywords = entry.get("keywords", [])
                overlap = _keyword_overlap(task_keywords, entry_keywords)
                if overlap >= 2:  # need at least 2 common keywords
                    scored.append((overlap, entry.get("compact_trace", "")))
    except Exception:
        return []

    # Sort by overlap descending, take top results
    scored.sort(key=lambda x: x[0], reverse=True)
    return [trace for _, trace in scored[:max_results]]


def build_few_shot_block(task):
    """Build a few-shot examples block for injection into the prompt.

    Returns a string to insert as a system message, or empty string.
    Max ~100 tokens to avoid bloating context.
    """
    examples = find_similar_examples(task)
    if not examples:
        return ""

    lines = ["Here are examples of similar tasks you handled successfully before:"]
    for i, trace in enumerate(examples, 1):
        # Truncate each trace to keep total size small
        lines.append(f"  Example {i}: {trace[:150]}")
    lines.append("Follow a similar approach if applicable.")

    return "\n".join(lines)


def count_few_shot():
    """Count number of few-shot examples stored."""
    if not os.path.isfile(FEW_SHOT_FILE):
        return 0
    try:
        with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


# ── Tier 2: DPO data collection ──────────────────────────────────────────

class DPOCollector:
    """Collects rejected/chosen pairs for DPO training.

    Usage:
        collector = DPOCollector(task, system_prompt)
        collector.activate()
        collector.add_rejected(bad_messages)  # can call multiple times
        collector.set_chosen(good_messages)
        saved = collector.save()
    """

    def __init__(self, task, system_prompt):
        self.task = task
        self.system_prompt = system_prompt
        self.active = False
        self._rejected = []  # list of message slices (each is a rejected attempt)
        self._chosen = None  # the final good message slice

    def activate(self):
        """Mark collector as active (user confirmed)."""
        self.active = True

    def add_rejected(self, messages_slice):
        """Add a rejected response (bad output + nudge)."""
        if not self.active:
            return
        # Deep copy to avoid mutation
        self._rejected.append([dict(m) for m in messages_slice])

    def set_chosen(self, messages_slice):
        """Set the chosen (good) response."""
        if not self.active:
            return
        self._chosen = [dict(m) for m in messages_slice]

    def save(self):
        """Save collected pairs to dpo_pairs.jsonl.

        Returns number of pairs saved.
        """
        if not self.active or not self._rejected or not self._chosen:
            return 0

        _ensure_data_dir()
        saved = 0

        try:
            with open(DPO_FILE, "a", encoding="utf-8") as f:
                for rejected in self._rejected:
                    entry = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "task": self.task[:200] if self.task else "",
                        "rejected": rejected,
                        "chosen": self._chosen,
                        "system_prompt": self.system_prompt[:500],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    saved += 1
        except Exception:
            return 0

        # Reset after saving
        self._rejected = []
        self._chosen = None
        return saved

    @staticmethod
    def count_pairs():
        """Count total DPO pairs stored."""
        if not os.path.isfile(DPO_FILE):
            return 0
        try:
            with open(DPO_FILE, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


def _extract_assistant_text(messages_slice):
    """Extract assistant's response text from a message slice."""
    for msg in reversed(messages_slice):
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            return msg["content"].strip()
    return ""


def find_dpo_guidance(task, max_results=2):
    """Find DPO pairs relevant to the current task.

    Returns list of (rejected_summary, chosen_summary) tuples.
    """
    if not os.path.isfile(DPO_FILE):
        return []

    task_keywords = _extract_keywords(task)
    if not task_keywords:
        return []

    scored = []
    try:
        with open(DPO_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_keywords = _extract_keywords(entry.get("task", ""))
                overlap = _keyword_overlap(task_keywords, entry_keywords)
                if overlap >= 2:
                    rejected_text = _extract_assistant_text(entry.get("rejected", []))
                    chosen_text = _extract_assistant_text(entry.get("chosen", []))
                    if rejected_text and chosen_text:
                        scored.append((overlap, rejected_text[:150], chosen_text[:150]))
    except Exception:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(r, c) for _, r, c in scored[:max_results]]


def build_dpo_block(task):
    """Build anti-pattern guidance from DPO data for injection into prompt.

    Returns a string to use as system message, or empty string.
    Pairs are injected immediately — no threshold, instant learning.
    """
    pairs = find_dpo_guidance(task)
    if not pairs:
        return ""

    lines = ["IMPORTANT — past corrections for similar tasks (learn from them):"]
    for i, (rejected, chosen) in enumerate(pairs, 1):
        lines.append(f"  WRONG {i}: {rejected}")
        lines.append(f"  RIGHT {i}: {chosen}")
    lines.append("Do NOT repeat the mistakes above.")
    return "\n".join(lines)


def show_learn_stats():
    """Return a formatted string with learning statistics."""
    fs_count = count_few_shot()
    dpo_count = DPOCollector.count_pairs()

    lines = [
        "Learning statistics:",
        f"  Few-shot examples: {fs_count}",
        f"  DPO pairs:         {dpo_count}",
    ]

    if dpo_count > 0:
        lines.append("  Status: Auto-learning active (injected as context)")
    else:
        lines.append("  Status: No DPO data yet")

    if fs_count > 0:
        lines.append(f"  Few-shot file: {FEW_SHOT_FILE}")
    if dpo_count > 0:
        lines.append(f"  DPO file:      {DPO_FILE}")

    return "\n".join(lines)
