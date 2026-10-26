"""Run the API with uvicorn, one or several worker processes.

Why not plain `uvicorn --workers N`: with several workers uvicorn binds the listening socket
itself as `socket.socket(family)`, i.e. with protocol 0. asyncio only sets TCP_NODELAY on a
connection whose socket says IPPROTO_TCP, so every worker connection kept Nagle's algorithm:
the response body waited for the client's delayed ACK, +~40 ms per request on Linux (measured in
the container: /model took 44 ms with 4 workers and 1.3 ms with 1; decision 027). Declaring the
same socket as TCP restores what single-worker mode already does.
"""

from __future__ import annotations

import socket

import uvicorn
from uvicorn.supervisors import Multiprocess

APP = "fraudplatform.serving.api:create_app"


class TcpConfig(uvicorn.Config):
    def bind_socket(self) -> socket.socket:
        sock = super().bind_socket()
        if sock.family in (socket.AF_INET, socket.AF_INET6) and sock.proto != socket.IPPROTO_TCP:
            return socket.socket(sock.family, sock.type, socket.IPPROTO_TCP, fileno=sock.detach())
        return sock


def run_api(host: str, port: int, workers: int) -> None:
    config = TcpConfig(APP, factory=True, host=host, port=port, workers=workers)
    if config.workers > 1:
        Multiprocess(config, sockets=[config.bind_socket()]).run()
    else:
        uvicorn.Server(config).run()
