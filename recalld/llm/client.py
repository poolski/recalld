from __future__ import annotations

import httpx


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat_url(self) -> str:
        base = self.base_url
        if base.endswith("/api/v1"):
            return f"{base}/chat"
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/api/v1/chat"

    def _parse_output(self, data: dict) -> str:
        output = data.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            message_parts = [
                item.get("content", "")
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
            ]
            if message_parts:
                return "\n".join(part for part in message_parts if part)
        raise ValueError("LM Studio response did not contain a usable message output")

    async def complete(self, system: str, user: str) -> str:
        """Send an LM Studio chat request. Returns the output text."""
        payload = {
            "model": self.model,
            "system_prompt": system,
            "input": user,
        }
        async with httpx.AsyncClient(verify=False, timeout=self.timeout) as client:
            resp = await client.post(self._chat_url(), json=payload)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_output(data)
