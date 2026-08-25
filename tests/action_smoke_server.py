#!/usr/bin/env python3
"""Deterministic loopback ASR fixture used by the composite-action smoke test."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(BaseHTTPRequestHandler):
    transcript = ""
    model = ""
    language = ""

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        expected = (
            self.path == "/v1/audio/transcriptions"
            and b'multipart/form-data' in self.headers.get("Content-Type", "").encode()
            and b'name="file"' in body
            and b'name="model"' in body
            and self.model.encode() in body
            and b'name="language"' in body
            and self.language.encode() in body
        )
        if not expected:
            payload = json.dumps({"error": "invalid smoke-test request"}).encode()
            self.send_response(422)
        else:
            payload = json.dumps({"text": self.transcript}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"Meta fixture: {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    _Handler.transcript = args.transcript
    _Handler.model = args.model
    _Handler.language = args.language
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    args.ready_file.write_text("ready\n", encoding="utf-8")
    print(f"Meta fixture listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
