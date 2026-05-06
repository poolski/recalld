from __future__ import annotations

import os
from typing import Callable, Optional

import httpx

class LLMRequestError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _api_v1_url(self, path: str) -> str:
        base = self.base_url
        if base.endswith("/api/v1"):
            return f"{base}{path}"
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/api/v1{path}"

    def _headers(self) -> dict[str, str]:
        token = os.getenv("LM_API_TOKEN")
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _chat_url(self) -> str:
        return self._api_v1_url("/chat")

    def _models_load_url(self) -> str:
        return self._api_v1_url("/models/load")

    def _chat_payload(self, system: str, user: str, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "system_prompt": system,
            "input": user,
            "stream": stream,
        }
        return payload

    def _extract_error_message(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
        except Exception:
            text = resp.text.strip()
            return text or f"LM Studio request failed with status {resp.status_code}"

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        text = resp.text.strip()
        return text or f"LM Studio request failed with status {resp.status_code}"

    def _parse_output(self, data: dict) -> str:
        if data.get("type") == "chat.end":
            result = data.get("result")
            if isinstance(result, dict):
                return self._parse_output(result)

        # Check for standard streaming output format (e.g., {"choices": [{"delta": {"content": "..."}}]})
        if "choices" in data:
            choices = data["choices"]
            if choices and "delta" in choices[0]:
                return choices[0]["delta"].get("content") or ""
            return ""
        
        # Fallback for original format
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
        return ""

    async def complete(self, system: str, user: str) -> str:
        """Send an LM Studio chat request. Returns the output text."""
        payload = self._chat_payload(system, user, stream=False)
        async with httpx.AsyncClient(verify=False, timeout=self.timeout) as client:
            resp = await client.post(self._chat_url(), json=payload, headers=self._headers())
            if resp.is_error:
                raise LLMRequestError(self._extract_error_message(resp))
            data = resp.json()
            return self._parse_output(data)

    async def load_model(self, context_length: int | None = None, echo_load_config: bool = True) -> dict:
        payload = {
            "model": self.model,
            "echo_load_config": echo_load_config,
        }
        if context_length is not None:
            payload["context_length"] = context_length
        async with httpx.AsyncClient(verify=False, timeout=self.timeout) as client:
            resp = await client.post(self._models_load_url(), json=payload, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def stream(
        self,
        system: str,
        user: str,
        event_cb: Optional[Callable[[str, dict], None]] = None,
    ):
        """Send a streaming LM Studio chat request. Yields partial content tokens."""
        import json
        payload = self._chat_payload(system, user, stream=True)
        async with httpx.AsyncClient(verify=False, timeout=self.timeout) as client:
            async with client.stream("POST", self._chat_url(), json=payload, headers=self._headers()) as resp:
                if resp.is_error:
                    raise LLMRequestError(self._extract_error_message(resp))
                event_type = None
                data_lines: list[str] = []
                yielded_message = False

                async for line in resp.aiter_lines():
                    if not line:
                        if event_type and data_lines:
                            data_str = "\n".join(data_lines)
                            if data_str != "[DONE]":
                                try:
                                    data = json.loads(data_str)
                                    if event_cb:
                                        event_cb(event_type, data)
                                    if event_type == "message.delta":
                                        token = data.get("content")
                                        if token:
                                            yielded_message = True
                                    elif event_type == "chat.end" and not yielded_message:
                                        token = self._parse_output(data)
                                    else:
                                        token = ""
                                    if token:
                                        yield token
                                except (json.JSONDecodeError, ValueError):
                                    pass
                        event_type = None
                        data_lines = []
                        continue

                    if line.startswith("event: "):
                        event_type = line[len("event: "):].strip()
                        continue

                    if line.startswith("data: "):
                        data_lines.append(line[len("data: "):].strip())
