"""A local, no-key mock of an OpenAI-compatible chat endpoint.

The tests use different URL paths to simulate common failure modes. This keeps
the test suite deterministic and independent of a real provider.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError):
            # The client may close while the server waits for another request.
            self.close_connection = True

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)

        if self.path == "/ok":
            self._json(
                200,
                {"choices": [{"message": {"content": "这是来自 mock 服务的回答。"}}]},
            )
        elif self.path == "/401":
            self._json(401, {"error": {"message": "invalid api key"}})
        elif self.path == "/400":
            self._json(400, {"error": {"message": "bad request"}})
        elif self.path == "/429":
            self._json(429, {"error": {"message": "rate limited"}})
        elif self.path == "/500":
            self._json(500, {"error": {"message": "server exploded"}})
        elif self.path == "/bad-json":
            self._raw(200, b"this is not json")
        elif self.path == "/missing-choices":
            self._json(200, {"id": "no-choices"})
        elif self.path == "/slow":
            time.sleep(0.3)
            self._json(
                200,
                {"choices": [{"message": {"content": "终于返回了。"}}]},
            )
        elif self.path == "/flaky":
            server = self.server
            server.flaky_count += 1
            if server.flaky_count <= server.fail_first:
                self._json(500, {"error": {"message": "temporary"}})
            else:
                self._json(
                    200,
                    {"choices": [{"message": {"content": "重试后成功。"}}]},
                )
        else:
            self._json(404, {"error": {"message": "unknown path"}})

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _raw(self, status: int, body: bytes) -> None:
        self._send(status, body, "text/plain")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
        ):
            # A client may have timed out and closed the socket already.
            pass

    def log_message(self, format: str, *args: Any) -> None:
        pass


class MockLLMServer:
    """Start/stop a local HTTP server on an ephemeral port."""

    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.fail_first = 0
        self._httpd.flaky_count = 0
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
        )

    def start(self) -> "MockLLMServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}{path}"

    def reset_flaky(self, fail_first: int) -> None:
        self._httpd.flaky_count = 0
        self._httpd.fail_first = fail_first

    @property
    def flaky_count(self) -> int:
        return self._httpd.flaky_count
