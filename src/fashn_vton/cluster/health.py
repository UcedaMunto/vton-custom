"""Minimal ``/healthz`` HTTP endpoint for the long-running workers.

The Kubernetes manifests use this endpoint for readiness *and* the
``Service vton-gpu-lb`` exposes it (plan section 7): a worker that cannot talk
to Redis/MinIO must not be marked ready.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


def _make_handler(payload: Callable[[], dict]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.startswith("/healthz") or self.path == "/":
                try:
                    body = json.dumps(payload(), default=str).encode("utf-8")
                    status = 200
                except Exception as exc:  # noqa: BLE001 - reported as JSON
                    body = json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
                    status = 503
            else:
                body, status = b"not found", 404
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # silence per-request logging
            return

    return Handler


def start_health_server(port: int, payload: Callable[[], dict], host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Start ``/healthz`` in a daemon thread and return the server."""
    server = ThreadingHTTPServer((host, int(port)), _make_handler(payload))
    threading.Thread(target=server.serve_forever, name="healthz", daemon=True).start()
    return server
