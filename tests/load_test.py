#!/usr/bin/env python3
"""
CLI load-test script for the Taxi Trip Duration Prediction API.

Usage:
    python tests/load_test.py --users 5 --requests-per-user 20
    python tests/load_test.py --users 100 --requests-per-user 10 --url http://localhost:8000
"""

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


API_URL = "http://localhost:8000"


def random_payload() -> dict:
    dow = random.randint(1, 7)
    return {
        "passenger_count": random.randint(0, 6),
        "trip_distance": round(random.uniform(0.1, 11.0), 2),
        "payment_type": random.choice([1, 2, 3, 4]),
        "pickup_hour": random.randint(0, 23),
        "pickup_dayofweek": dow,
        "pickup_month": random.randint(1, 12),
        "is_weekend": 1 if dow in (1, 7) else 0,
    }


def send_request(session: requests.Session, url: str, model_type: str) -> dict:
    payload = random_payload()
    start = time.perf_counter()
    try:
        resp = session.post(f"{url}/predict", json=payload, params={"model_type": model_type}, timeout=30)
        latency = time.perf_counter() - start
        return {"latency": latency, "status": resp.status_code, "success": resp.status_code == 200}
    except Exception as e:
        latency = time.perf_counter() - start
        return {"latency": latency, "status": 0, "success": False, "error": str(e)}


def percentile(data: list[float], p: float) -> float:
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def main():
    parser = argparse.ArgumentParser(description="Load test the taxi prediction API")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users/threads")
    parser.add_argument("--requests-per-user", type=int, default=10, help="Requests each user sends")
    parser.add_argument("--url", type=str, default=API_URL, help="Base URL of the API")
    parser.add_argument("--model-type", type=str, default="tflite", choices=["keras", "tflite"],
                        help="Model to use: 'keras' (full NN) or 'tflite' (pruned)")
    args = parser.parse_args()

    total = args.users * args.requests_per_user
    print(f"\nLoad Test: {args.users} concurrent users × {args.requests_per_user} requests = {total} total requests")
    print(f"Target: {args.url}  |  Model: {args.model_type}\n")

    session = requests.Session()
    results: list[dict] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = [pool.submit(send_request, session, args.url, args.model_type) for _ in range(total)]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % max(1, total // 10) == 0:
                print(f"   Progress: {i}/{total} ({i/total*100:.0f}%)")

    wall_time = time.perf_counter() - wall_start
    latencies = sorted(r["latency"] for r in results)
    success = sum(1 for r in results if r["success"])
    errors = total - success

    print("\n" + "=" * 55)
    print("LOAD TEST RESULTS")
    print("=" * 55)
    print(f"  Total requests        : {total}")
    print(f"  Successful            : {success}")
    print(f"  Failed                : {errors}")
    print(f"  Error rate            : {errors/total*100:.2f}%")
    print(f"  Total wall time       : {wall_time:.2f}s")
    print(f"  Throughput            : {total/wall_time:.2f} req/s")
    print("-" * 55)
    print(f"  Mean latency          : {statistics.mean(latencies)*1000:.2f} ms")
    print(f"  Median (P50) latency  : {statistics.median(latencies)*1000:.2f} ms")
    print(f"  P95 latency           : {percentile(latencies, 95)*1000:.2f} ms")
    print(f"  P99 latency           : {percentile(latencies, 99)*1000:.2f} ms")
    print(f"  Min latency           : {min(latencies)*1000:.2f} ms")
    print(f"  Max latency           : {max(latencies)*1000:.2f} ms")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
