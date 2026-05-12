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
      - a parsed dict if `schema` is given (request `format="json"`, then
        json.loads the response; pydantic validation downstream enforces
        the actual schema shape).
      - the raw string otherwise.

    Why not Ollama's `format=<JSON schema>` constrained-decoding mode?
    Newer models (qwen3.6:27b confirmed; likely others) return empty
    bodies for schema-mode requests — they don't honor it. `format="json"`
    has universal model support and gives the same end result for us
    because we always re-validate the parsed JSON against our pydantic
    models anyway. `schema` here is accepted for API stability and
    docs; we don't actually pass it to Ollama.

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
            # Disable thinking-mode for reasoning models like qwen3.6:27b.
            # With format='json' constraining output, thinking tokens get
            # suppressed and the model produces empty responses. We don't
            # need deliberation for fact extraction or placement — we want
            # direct JSON output. The parameter is silently ignored by
            # non-thinking models (e.g. qwen3:32b), so this is safe to
            # always send.
            "think": False,
        }
        if system is not None:
            body["system"] = system
        if num_ctx is not None:
            body["options"]["num_ctx"] = num_ctx
        # `schema` is accepted on the API for caller-side docs but isn't
        # forwarded as Ollama's constrained-decoding format — qwen3.6:27b
        # and similar newer models return empty bodies in that mode.
        # format='json' has universal model support; pydantic validation
        # downstream enforces our actual schema shape.
        if schema is not None:
            body["format"] = "json"

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
                f"Ollama returned invalid JSON: {response_text[:500]}"
            ) from e
