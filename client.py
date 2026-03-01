import json
import urllib.request
from config import BASE_URL, API_KEY, MODEL, SUMMARY_PROMPT


class LMClient:
    def __init__(self):
        self.model = MODEL
        self.base_url = BASE_URL

    def _api_base(self):
        """Get base URL without /v1 suffix."""
        return self.base_url.replace("/v1", "")

    def list_models(self):
        """Fetch only currently loaded models from LM Studio.

        Returns list of (model_key, context_length) tuples.
        context_length is the actual loaded context size, or 0 if unknown.
        """
        r = urllib.request.urlopen(f"{self._api_base()}/api/v1/models")
        data = json.loads(r.read())
        loaded = []
        for m in data.get("models", []):
            instances = m.get("loaded_instances")
            if instances:
                ctx = 0
                if instances[0].get("config"):
                    ctx = instances[0]["config"].get("context_length", 0)
                if not ctx:
                    ctx = m.get("max_context_length", 0)
                loaded.append((m["key"], ctx))
        return loaded

    def chat(self, messages, tools=None):
        """Send a chat request and return the parsed response message."""
        body = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]

    def summarize(self, messages):
        """Summarize a list of messages into a short text via the model.

        Truncates long messages and caps total size to avoid 400 errors
        from the LLM backend when the conversation is too large.
        """
        MAX_SUMMARY_CHARS = 12000  # keep summarization request small

        lines = []
        total = 0
        for m in messages:
            content = m.get("content", "")
            # Truncate individual tool results / long messages
            if len(content) > 500:
                content = content[:500] + "..."
            line = f"{m['role']}: {content}"
            if total + len(line) > MAX_SUMMARY_CHARS:
                lines.append("... (earlier messages truncated)")
                break
            lines.append(line)
            total += len(line)

        conversation = "\n".join(lines)
        summary_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": conversation},
        ]
        result = self.chat(summary_messages)
        return result.get("content", "")

    def stream_chat(self, messages, tools=None):
        """Send a streaming chat request. Yields parsed SSE chunks."""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        )
        resp = urllib.request.urlopen(req)

        # Read SSE stream line by line
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    yield chunk
                except json.JSONDecodeError:
                    continue
