"""CLI for local smoke and stdlib HTTP serving of the inference service."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from pathlib import Path

logger = logging.getLogger("streamcompiler.server.cli")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _listen_address(raw: str) -> tuple[str, int]:
    host, separator, port_raw = raw.rpartition(":")
    if not separator or not host or not port_raw:
        raise argparse.ArgumentTypeError("expected HOST:PORT")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return host, port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamcompiler-serve")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument(
        "--listen",
        metavar="HOST:PORT",
        type=_listen_address,
        help="serve HTTP (health/ready/metrics/infer) on HOST:PORT",
    )
    parser.add_argument(
        "--devices",
        metavar="ID[,ID...]",
        help="comma-separated device worker ids (virtual labels or GPU ordinals)",
    )
    parser.add_argument("--artifact", type=Path, help="compiled artifact directory to load")
    parser.add_argument("--model-id", default="default", help="served model identifier")
    parser.add_argument("--concurrency", type=_positive_int, help="per-model concurrency limit")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow network serving without a loaded artifact (health diagnostics only)",
    )
    args = parser.parse_args(argv)

    if args.listen and args.artifact is None and not args.allow_empty:
        parser.error("--listen requires --artifact (or explicit --allow-empty for diagnostics)")

    from streamcompiler.runtime.module import load_compiled
    from streamcompiler.serve import InferenceService, ServiceConfig
    from streamcompiler.serve.http import HttpServer

    device_workers = None
    if args.devices:
        from streamcompiler.runtime.device_workers import DeviceWorkerSupervisor

        ids = [p.strip() for p in args.devices.split(",") if p.strip()]
        device_workers = DeviceWorkerSupervisor(device_ids=ids)

    config = ServiceConfig.from_env()
    svc = InferenceService(config=config, device_workers=device_workers)
    try:
        if args.artifact is not None:
            compiled = load_compiled(args.artifact)
            svc.models.load(
                args.model_id,
                compiled,
                concurrency_limit=args.concurrency or config.default_concurrency,
            )
        svc.start()
        if args.health:
            print(json.dumps(svc.health()))
            return 0
        if args.metrics:
            sys.stdout.write(svc.metrics_prometheus())
            return 0
        if args.listen:
            host, port = args.listen
            http = HttpServer(svc, host=host, port=port)
            stop_requested = threading.Event()

            def request_stop(signum: int, _frame: object) -> None:
                logger.info("received signal %s; stopping service", signum)
                stop_requested.set()

            previous = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)}
            try:
                http.start(background=True)
                stop_requested.wait()
            finally:
                http.stop()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            return 0
        print(json.dumps({"ready": svc.readiness(), "health": svc.health()}))
        return 0
    finally:
        svc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
