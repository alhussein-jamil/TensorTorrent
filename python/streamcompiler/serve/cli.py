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
    parser.add_argument(
        "--devices",
        metavar="ID[,ID...]",
        help="comma-separated device worker ids (virtual labels or GPU ordinals)",
    )
    args = parser.parse_args(argv)

    from streamcompiler.serve import InferenceService
    from streamcompiler.serve.http import HttpServer

    device_workers = None
    if args.devices:
        from streamcompiler.runtime.device_workers import DeviceWorkerSupervisor

        ids = [p.strip() for p in args.devices.split(",") if p.strip()]
        device_workers = DeviceWorkerSupervisor(device_ids=ids)

    svc = InferenceService(device_workers=device_workers)
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
