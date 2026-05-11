from __future__ import annotations

import json
import logging
from typing import Any

import httpx

log = logging.getLogger("poptrivia.prep.llm")


class OllamaUnavailable(RuntimeError):
    """Raised when Ollama is unreachable or returns a transport-level error.

    The prep queue worker uses this to back off rather than fail the job —
    the user's main PC is expected to be off some of the time.
    """


class OllamaError(RuntimeError):
    """Raised for Ollama responses we got but couldn't parse / use."""


class OllamaClient:
    """Minimal Ollama /api/generate wrapper.

    `generate(prompt, schema=None)` returns:
      - a parsed dict if schema is given (JSON mode + validation)
      - the raw string otherwise

    On connection refused / DNS failure / timeout: raises OllamaUnavailable.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ):
        if not base_url:
            raise ValueError("OllamaClient requires a base URL")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        """Lightweight reachability probe — /api/tags returns the model list."""
        try:
            r = await self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            return r.status_code == 200
        except httpx.HTTPError as e:
            log.debug("Ollama health probe failed: %s", e)
            return False

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
    ) -> dict[str, Any] | str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system is not None:
            body["system"] = system
        if num_ctx is not None:
            body["options"]["num_ctx"] = num_ctx
        if schema is not None:
            # Ollama accepts a JSON schema directly here.
            body["format"] = schema

        try:
            r = await self._client.post(
                f"{self.base_url}/api/generate", json=body, timeout=self.timeout
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            raise OllamaUnavailable(f"Ollama unreachable: {e}") from e
        except httpx.HTTPError as e:
            raise OllamaUnavailable(f"Ollama transport error: {e}") from e

        if r.status_code != 200:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

        try:
            payload = r.json()
        except ValueError as e:
            raise OllamaError(f"Ollama returned non-JSON: {r.text[:500]}") from e

        response_text = payload.get("response")
        if not isinstance(response_text, str):
            raise OllamaError(f"Ollama response missing 'response' field: {payload!r}")

        if schema is None:
            return response_text

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise OllamaError(
                f"Ollama returned invalid JSON despite schema: {response_text[:500]}"
            ) from e
