"""CLI for local smoke and stdlib HTTP serving of the inference service."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamcompiler-serve")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument(
        "--listen",
        metavar="HOST:PORT",
        help="serve HTTP (health/ready/metrics/infer) on HOST:PORT",
    )
    args = parser.parse_args(argv)

    from server import InferenceService
    from server.http import HttpServer

    svc = InferenceService()
    svc.start()
    try:
        if args.health:
            print(json.dumps(svc.health()))
            return 0
        if args.metrics:
            sys.stdout.write(svc.metrics_prometheus())
            return 0
        if args.listen:
            host, _, port_s = args.listen.partition(":")
            if not host or not port_s:
                parser.error("--listen expects HOST:PORT")
            http = HttpServer(svc, host=host, port=int(port_s))
            http.start(background=False)
            return 0
        print(json.dumps({"ready": svc.readiness(), "health": svc.health()}))
        return 0
    finally:
        svc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
