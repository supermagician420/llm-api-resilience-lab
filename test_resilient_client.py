"""Tests for the retry/error-classification behavior.

Run from this directory with:

    python -m unittest -v
"""

from __future__ import annotations

import unittest
from typing import List, Optional

from mock_server import MockLLMServer
from resilient_client import (
    AuthenticationError,
    BadRequestError,
    ConfigurationError,
    LLMTimeoutError,
    RateLimitError,
    ResilientLLMClient,
    ResponseError,
    RetryPolicy,
    ValidationError,
    user_message,
)


class ResilientClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = MockLLMServer().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def client(
        self,
        path: str = "/ok",
        *,
        max_attempts: int = 1,
        timeout: float = 1.0,
        api_key: str = "test-key",
        sleeps: Optional[List[float]] = None,
    ) -> ResilientLLMClient:
        policy = RetryPolicy(
            max_attempts=max_attempts,
            base_delay=0.01,
            max_delay=0.02,
        )
        sleep = sleeps.append if sleeps is not None else (lambda _: None)
        return ResilientLLMClient(
            self.server.url(path),
            api_key,
            "test-model",
            timeout=timeout,
            policy=policy,
            sleep=sleep,
        )

    def test_success_returns_assistant_text(self) -> None:
        self.assertEqual(
            self.client().chat("报销需要哪些材料？"),
            "这是来自 mock 服务的回答。",
        )

    def test_missing_api_key_fails_before_network(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.client(api_key="")

    def test_empty_question_is_rejected_locally(self) -> None:
        with self.assertRaises(ValidationError):
            self.client().chat("   ")

    def test_401_is_not_retried(self) -> None:
        with self.assertRaises(AuthenticationError) as ctx:
            self.client("/401", max_attempts=3).chat("你好")
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.attempts, 1)
        self.assertFalse(ctx.exception.retryable)

    def test_400_is_not_retried(self) -> None:
        with self.assertRaises(BadRequestError) as ctx:
            self.client("/400", max_attempts=3).chat("你好")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.attempts, 1)

    def test_429_retries_then_raises_rate_limit(self) -> None:
        sleeps: List[float] = []
        with self.assertRaises(RateLimitError) as ctx:
            self.client("/429", max_attempts=2, sleeps=sleeps).chat("你好")
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(sleeps, [0.01])

    def test_500_is_retried_then_succeeds(self) -> None:
        self.server.reset_flaky(fail_first=1)
        answer = self.client("/flaky", max_attempts=2).chat("你好")
        self.assertEqual(answer, "重试后成功。")
        self.assertEqual(self.server.flaky_count, 2)

    def test_bad_json_retries_then_raises_response_error(self) -> None:
        sleeps: List[float] = []
        with self.assertRaises(ResponseError) as ctx:
            self.client("/bad-json", max_attempts=2, sleeps=sleeps).chat("你好")
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(sleeps, [0.01])

    def test_missing_choices_is_not_retried(self) -> None:
        with self.assertRaises(ResponseError) as ctx:
            self.client("/missing-choices", max_attempts=3).chat("你好")
        self.assertEqual(ctx.exception.attempts, 1)
        self.assertFalse(ctx.exception.retryable)

    def test_timeout_retries_then_raises_timeout(self) -> None:
        sleeps: List[float] = []
        with self.assertRaises(LLMTimeoutError) as ctx:
            self.client(
                "/slow",
                max_attempts=2,
                timeout=0.05,
                sleeps=sleeps,
            ).chat("你好")
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(sleeps, [0.01])

    def test_user_message_hides_stack_trace(self) -> None:
        message = user_message(AuthenticationError("boom", status=401))
        self.assertIn("LLM_API_KEY", message)
        self.assertNotIn("boom", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
