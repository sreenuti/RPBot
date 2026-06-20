#!/usr/bin/env python3
"""Launch the RealPage Message Agent demo UI."""

import socket
import sys

import uvicorn


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    port = 8080
    if _port_in_use(port):
        print(
            f"Port {port} is already in use. Stop the other server (Ctrl+C) and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Starting demo UI at http://127.0.0.1:{port}")
    uvicorn.run(
        "ui.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )
