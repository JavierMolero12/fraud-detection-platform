"""Latency benchmark of the real API: uvicorn + HTTP + online features + model + logging.

Python inference is GIL-bound, so throughput comes from processes, not threads. The benchmark
starts N independent single-worker uvicorn servers on consecutive ports and its client threads
spread the requests over them, as a load balancer would. (uvicorn --workers N would share one
socket instead, but on Windows some workers fail to listen on it: WinError 10022.)

Replays stored production-period payments as new requests (ids prefixed with the run id):
first one at a time against one server (latency), then from several client threads against
1..N servers (throughput). Server-side latency comes from scoring.predictions.latency_ms. The
benchmark rows are deleted afterwards unless keep=True, so they never reach monitoring.

If a batch scoring CSV is given, every API probability and decision is compared with the batch
output for the same transaction: both paths must agree.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psycopg

log = logging.getLogger(__name__)

SAMPLE = """
SELECT transaction_id, customer_id, card_id, merchant_id, ts, amount, currency_code,
       country_code, payment_method, device_id
FROM core.transactions
WHERE ts >= %s AND ts < %s
ORDER BY md5(transaction_id || %s)
LIMIT %s
"""


def _payload(row: tuple, run: str) -> dict:
    txid, customer, card, merchant, ts, amount, currency, country, method, device = row
    body = {
        "transaction_id": f"bench-{run}-{txid}",
        "customer_id": customer,
        "card_id": card,
        "merchant_id": merchant,
        "timestamp": ts.isoformat(),
        "amount": float(amount),
        "currency": currency,
        "country": country,
        "payment_method": method,
    }
    if device is not None:
        body["device_id"] = device
    return body


def _percentiles(ms) -> dict[str, float]:
    ms = np.asarray(ms, dtype=float)
    return {
        "p50": round(float(np.percentile(ms, 50)), 1),
        "p95": round(float(np.percentile(ms, 95)), 1),
        "p99": round(float(np.percentile(ms, 99)), 1),
        "max": round(float(ms.max()), 1),
        "mean": round(float(ms.mean()), 1),
    }


def _median_over(repetitions: list[dict]) -> dict:
    """Median of each reported number over the repetitions (same structure, one level)."""

    def med(values) -> float:
        return round(float(np.median(values)), 1)

    seq = [r["sequential"] for r in repetitions]
    return {
        "sequential": {
            kind: {q: med([s[kind][q] for s in seq]) for q in seq[0][kind]}
            for kind in ("client_ms", "server_ms")
        },
        "concurrent": [
            {
                "server_processes": runs[0]["server_processes"],
                "requests_per_second": med([c["requests_per_second"] for c in runs]),
                "server_ms": {
                    q: med([c["server_ms"][q] for c in runs]) for q in runs[0]["server_ms"]
                },
            }
            for runs in zip(*[r["concurrent"] for r in repetitions], strict=True)
        ],
    }


def _start(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "fraudplatform.serving.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
    )


def _stop(servers: list[subprocess.Popen]) -> None:
    for server in servers:
        server.terminate()
    for server in servers:
        server.wait(timeout=30)


def _serve(ports: list[int]) -> list[subprocess.Popen]:
    """One API process per port (as deployed), so the client never shares their GIL."""
    servers = [_start(port) for port in ports]
    deadline = time.time() + 180
    pending = set(ports)
    try:
        while pending:
            if time.time() > deadline:
                raise RuntimeError(f"API on ports {sorted(pending)} not healthy within 180 s")
            if any(server.poll() is not None for server in servers):
                raise RuntimeError("an API process exited during start-up")
            for port in list(pending):
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                        pending.discard(port)
                except httpx.TransportError:
                    pass
            time.sleep(0.5)
    except BaseException:
        _stop(servers)
        raise
    return servers


def _concurrent(payloads: list[dict], ports: list[int], threads: int) -> tuple[list, float]:
    """Client thread i sends to server i % len(ports), on its own keep-alive connection."""
    local = threading.local()
    ids = iter(range(threads))
    lock = threading.Lock()

    def post(body: dict) -> dict:
        if not hasattr(local, "client"):
            with lock:
                i = next(ids)
            local.client = httpx.Client(timeout=60)
            local.url = f"http://127.0.0.1:{ports[i % len(ports)]}/predict"
        r = local.client.post(local.url, json=body)
        r.raise_for_status()
        return r.json()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(post, payloads))
    return results, time.perf_counter() - t0


def run_bench(
    conn: psycopg.Connection,
    start: date,
    end: date,
    n_sequential: int,
    n_concurrent: int,
    threads: int,
    processes: list[int],
    port: int,
    output: Path,
    batch_csv: Path | None = None,
    keep: bool = False,
    repeats: int = 1,
) -> dict:
    """On a shared machine one run can be distorted (a checkpoint, another busy program), so
    the measurement can be repeated on the same servers: each repetition is reported, plus the
    median over repetitions."""
    run = uuid.uuid4().hex[:8]
    per_repeat = n_sequential + n_concurrent * len(processes)
    rows = conn.execute(SAMPLE, (start, end, run, per_repeat * repeats)).fetchall()
    payloads = [_payload(r, run) for r in rows]
    ports = [port + i for i in range(max(processes))]
    responses: dict[str, dict] = {}
    measured = []  # per repetition: (sequential payloads, client ms, [(n, chunk, seconds)])

    servers = _serve(ports)
    try:
        for p in ports:  # warm-up: pool connections, first predictions
            with httpx.Client(timeout=60) as client:
                for body in payloads[:20]:
                    warm = {**body, "transaction_id": f"{body['transaction_id']}-w{p}"}
                    client.post(f"http://127.0.0.1:{p}/predict", json=warm).raise_for_status()

        for rep in range(repeats):
            batch = payloads[rep * per_repeat : (rep + 1) * per_repeat]
            sequential = batch[:n_sequential]
            client_ms = []
            with httpx.Client(timeout=60) as client:
                for body in sequential:
                    t0 = time.perf_counter()
                    r = client.post(f"http://127.0.0.1:{ports[0]}/predict", json=body)
                    client_ms.append((time.perf_counter() - t0) * 1000)
                    r.raise_for_status()
                    responses[body["transaction_id"]] = r.json()
            concurrent_runs = []
            for k, n in enumerate(processes):
                first = n_sequential + k * n_concurrent
                chunk = batch[first : first + n_concurrent]
                results, seconds = _concurrent(chunk, ports[:n], threads)
                responses |= {r["transaction_id"]: r for r in results}
                concurrent_runs.append((n, chunk, seconds))
            measured.append((sequential, client_ms, concurrent_runs))
    finally:
        _stop(servers)

    server_ms = dict(
        conn.execute(
            "SELECT transaction_id, latency_ms FROM scoring.predictions "
            "WHERE source = 'api' AND transaction_id LIKE %s",
            (f"bench-{run}-%",),
        ).fetchall()
    )
    repetitions = [
        {
            "sequential": {
                "requests": len(sequential),
                "server_processes": 1,
                "client_ms": _percentiles(client_ms),
                "server_ms": _percentiles([server_ms[b["transaction_id"]] for b in sequential]),
            },
            "concurrent": [
                {
                    "requests": len(chunk),
                    "client_threads": threads,
                    "server_processes": n,
                    "seconds": round(seconds, 2),
                    "requests_per_second": round(len(chunk) / seconds, 1),
                    "server_ms": _percentiles([server_ms[b["transaction_id"]] for b in chunk]),
                }
                for n, chunk, seconds in concurrent_runs
            ],
        }
        for sequential, client_ms, concurrent_runs in measured
    ]
    result = {
        "run": run,
        "period": [str(start), str(end)],
        "repeats": repeats,
        "median_over_repeats": _median_over(repetitions),
        "repetitions": repetitions,
        "decisions": pd.Series([r["decision"] for r in responses.values()])
        .value_counts()
        .to_dict(),
    }
    if batch_csv is not None:
        result["agreement_with_batch"] = _compare_with_batch(responses, run, batch_csv)

    if not keep:
        conn.execute(
            "DELETE FROM scoring.predictions WHERE source = 'api' AND transaction_id LIKE %s",
            (f"bench-{run}-%",),
        )
        conn.commit()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    return result


def run_profile(
    conn: psycopg.Connection, start: date, end: date, n: int, output: Path, warmup: int = 20
) -> dict:
    """Milliseconds per step of score_payment, in this process, one payment at a time.

    Instruments the real code path (no HTTP). `conn` must be in autocommit mode. On a laptop the
    absolute numbers depend on the CPU power state (a loop that mostly waits on the database
    runs at a lower clock); use the shares, and the benchmark for end-to-end latency.
    """
    from threadpoolctl import threadpool_limits

    from fraudplatform.config import load_feature_config, load_serving_config
    from fraudplatform.serving.online import score_payment
    from fraudplatform.serving.online_features import IncomingTransaction, OnlineFeatureBuilder
    from fraudplatform.serving.scorer import Bundle, Scorer

    threadpool_limits(limits=1, user_api="openmp")  # as in the API
    serving = load_serving_config()
    scorer = Scorer(Bundle.load(serving.bundle_dir), serving.risk_levels)
    features = OnlineFeatureBuilder(load_feature_config())
    run = uuid.uuid4().hex[:8]
    rows = conn.execute(SAMPLE, (start, end, run, n + warmup)).fetchall()
    steps: list[dict[str, float]] = []
    try:
        for txid, *values in rows:
            txn = IncomingTransaction(*values[:4], float(values[4]), *values[5:])
            timings: dict[str, float] = {}
            score_payment(conn, scorer, features, f"profile-{run}-{txid}", txn, timings=timings)
            steps.append(timings)
    finally:
        conn.execute(
            "DELETE FROM scoring.predictions WHERE source = 'api' AND transaction_id LIKE %s",
            (f"profile-{run}-%",),
        )
    table = pd.DataFrame(steps[warmup:])
    table["total"] = table.sum(axis=1)
    result = {
        "payments": len(table),
        "period": [str(start), str(end)],
        "steps_ms": {
            col: {
                "p50": round(float(table[col].median()), 2),
                "p95": round(float(table[col].quantile(0.95)), 2),
                "share_of_mean_total": round(float(table[col].mean() / table["total"].mean()), 3),
            }
            for col in table.columns
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    return result


def _compare_with_batch(responses: dict[str, dict], run: str, batch_csv: Path) -> dict:
    api = pd.DataFrame(responses.values())
    api["transaction_id"] = api["transaction_id"].str.removeprefix(f"bench-{run}-")
    batch = pd.read_csv(
        batch_csv,
        usecols=["transaction_id", "fraud_probability", "decision"],
        float_precision="round_trip",  # the default parser can be 1 ulp off
    )
    both = api.merge(batch, on="transaction_id", suffixes=("_api", "_batch"))
    diff = (both["fraud_probability_api"] - both["fraud_probability_batch"]).abs()
    return {
        "compared": int(len(both)),
        "max_abs_probability_difference": float(diff.max()),
        "same_decision": int((both["decision_api"] == both["decision_batch"]).sum()),
    }
