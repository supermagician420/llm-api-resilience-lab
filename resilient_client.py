"""A tiny, dependency-free client for calling an LLM chat API safely.

The module focuses on one question: what should happen when the remote API is
slow, misconfigured, rate-limited, unavailable, or returns an unusable body?
It classifies errors, retries only the transient ones, and exposes a short
message that a CLI can show without leaking stack traces.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Optional


class LLMError(Exception):
    """Base error with fields a UI can branch on without parsing text."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        retryable: bool = False,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.attempts = attempts

    def __str__(self) -> str:
        return self.message


class ConfigurationError(LLMError):
    """The local API URL, key, model, or timeout is missing/invalid."""


class ValidationError(LLMError):
    """The caller supplied an empty or unusable question."""


class AuthenticationError(LLMError):
    """The server rejected the API key (401 or 403)."""


class BadRequestError(LLMError):
    """The server rejected the request body (4xx other than auth/429)."""


class RateLimitError(LLMError):
    """The server asked us to slow down (429)."""


class ServerError(LLMError):
    """The server failed with a 5xx response."""


class NetworkError(LLMError):
    """A connection could not be established or was interrupted."""


class LLMTimeoutError(LLMError):
    """The request exceeded the client-side timeout."""


class ResponseError(LLMError):
    """The HTTP call succeeded but the body was not usable."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff, kept deliberately simple."""

    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")

    def delay_for(self, attempt: int) -> float:
        """Return the delay after a failed 1-based attempt."""
        return min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))


class ResilientLLMClient:
    """Call an OpenAI-compatible chat-completions endpoint with retries."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 10.0,
        policy: Optional[RetryPolicy] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_url or not api_url.strip():
            raise ConfigurationError("LLM API URL 未配置，请设置 LLM_API_URL。")
        if not api_key or not api_key.strip():
            raise ConfigurationError("LLM API Key 未配置，请设置 LLM_API_KEY。")
        if not model or not model.strip():
            raise ConfigurationError("LLM 模型名未配置，请设置 LLM_MODEL。")
        if timeout <= 0:
            raise ConfigurationError("timeout 必须大于 0。")

        self.api_url = api_url.strip()
        parsed = urllib.parse.urlsplit(self.api_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ConfigurationError(
                "LLM_API_URL 必须是合法的 http(s) 地址。"
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise ConfigurationError("LLM_API_URL 的端口不合法。") from exc

        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = port
        self._path = parsed.path or "/"
        if parsed.query:
            self._path += "?" + parsed.query
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout
        self.policy = policy or RetryPolicy()
        self._sleep = sleep

    def chat(self, question: str) -> str:
        """Return assistant text, or raise an LLMError subclass."""
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValidationError("问题不能为空，请输入一个具体问题。")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": cleaned}],
                "temperature": 0.2,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        last_error: Optional[LLMError] = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                return self._call_once(payload)
            except LLMError as exc:
                exc.attempts = attempt
                last_error = exc
                if not exc.retryable or attempt == self.policy.max_attempts:
                    raise
                self._sleep(self.policy.delay_for(attempt))

        raise last_error or LLMError("AI 服务调用失败。")

    def _call_once(self, payload: bytes) -> str:
        connection_cls = (
            http.client.HTTPSConnection
            if self._scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            self._host,
            self._port,
            timeout=self.timeout,
        )

        try:
            connection.request(
                "POST",
                self._path,
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            response = connection.getresponse()
            status = response.status
            if not 200 <= status < 300:
                self._raise_for_status(status)
            raw = response.read()
        except (socket.timeout, TimeoutError) as exc:
            raise LLMTimeoutError(
                "AI 服务响应超时，请稍后重试。",
                retryable=True,
            ) from exc
        except http.client.IncompleteRead as exc:
            raise ResponseError(
                "AI 服务返回的响应不完整。",
                retryable=True,
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise NetworkError(
                "无法连接 AI 服务，请检查网络或 API 地址。",
                retryable=True,
            ) from exc
        finally:
            connection.close()

        try:
            data: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponseError(
                "AI 服务返回了无法解析的内容。",
                retryable=True,
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ResponseError(
                "AI 服务返回结构异常，缺少 choices/message/content。",
                retryable=False,
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise ResponseError("AI 服务返回了空回答。", retryable=False)
        return content.strip()

    @staticmethod
    def _raise_for_status(status: int) -> None:
        if status in (401, 403):
            raise AuthenticationError(
                "AI 服务认证失败，请检查 API Key 是否有效。",
                status=status,
            )
        if status == 429:
            raise RateLimitError(
                "AI 服务请求过于频繁（429），请稍后重试。",
                status=status,
                retryable=True,
            )
        if 500 <= status < 600:
            raise ServerError(
                f"AI 服务暂时不可用（{status}），请稍后重试。",
                status=status,
                retryable=True,
            )
        if 400 <= status < 500:
            raise BadRequestError(
                f"AI 服务拒绝了请求（{status}），请检查输入或模型配置。",
                status=status,
            )
        raise LLMError(
            f"AI 服务返回了未预期的状态码（{status}）。",
            status=status,
        )


def user_message(error: BaseException) -> str:
    """Map any exception to a short message suitable for a CLI/user."""
    if isinstance(error, ConfigurationError):
        return "配置不完整：请先设置 LLM_API_URL、LLM_API_KEY 和 LLM_MODEL。"
    if isinstance(error, ValidationError):
        return "输入无效：请粘贴一个非空的问题。"
    if isinstance(error, AuthenticationError):
        return "认证失败：请检查 LLM_API_KEY，或确认该 Key 是否有权限。"
    if isinstance(error, RateLimitError):
        return "请求太频繁：请等待一会儿再试，或降低并发。"
    if isinstance(error, ServerError):
        return "服务端暂时不可用：请稍后重试；如果持续失败，请查看服务状态。"
    if isinstance(error, LLMTimeoutError):
        return "请求超时：网络可能较慢，请稍后重试或调大 timeout。"
    if isinstance(error, NetworkError):
        return "网络错误：请检查网络连接和 API 地址。"
    if isinstance(error, ResponseError):
        return "响应异常：AI 服务返回了无法使用的数据，请稍后重试。"
    if isinstance(error, BadRequestError):
        return "请求被拒绝：请检查问题内容、模型名或接口参数。"
    return "发生未知错误：请查看日志后重试。"


__all__ = [
    "AuthenticationError",
    "BadRequestError",
    "ConfigurationError",
    "LLMError",
    "LLMTimeoutError",
    "NetworkError",
    "RateLimitError",
    "ResilientLLMClient",
    "ResponseError",
    "RetryPolicy",
    "ServerError",
    "ValidationError",
    "user_message",
]
