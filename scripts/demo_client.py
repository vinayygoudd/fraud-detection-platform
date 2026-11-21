"""Interactive demonstration client exercising /health, /model-info, and /predict endpoints."""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo client for the Fraud Detection FastAPI service.")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running FastAPI service.",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=15.0) as client:
        print("=== 1. GET /health ===")
        health = client.get(f"{base}/health")
        print(f"Status: {health.status_code}")
        print(json.dumps(health.json(), indent=2))

        print("\n=== 2. GET /model-info ===")
        info = client.get(f"{base}/model-info")
        print(f"Status: {info.status_code}")
        if info.status_code == 200:
            print(json.dumps(info.json(), indent=2))
        else:
            print(info.text)
            sys.exit(1)

        sample_requests = [
            {
                "label": "Low-risk small PAYMENT (cold-start or normal)",
                "payload": {
                    "transaction_id": "demo-tx-001",
                    "customer_id": "C_DEMO_USER_01",
                    "amount": 45.50,
                    "tx_type": "PAYMENT",
                    "step": 745,
                },
            },
            {
                "label": "High-value TRANSFER (elevated risk profile)",
                "payload": {
                    "transaction_id": "demo-tx-002",
                    "customer_id": "C_DEMO_USER_01",
                    "amount": 285000.00,
                    "tx_type": "TRANSFER",
                    "step": 746,
                },
            },
            {
                "label": "Idempotent retry of demo-tx-002",
                "payload": {
                    "transaction_id": "demo-tx-002",
                    "customer_id": "C_DEMO_USER_01",
                    "amount": 285000.00,
                    "tx_type": "TRANSFER",
                    "step": 746,
                },
            },
        ]

        for idx, item in enumerate(sample_requests, start=3):
            print(f"\n=== {idx}. POST /predict — {item['label']} ===")
            resp = client.post(f"{base}/predict", json=item["payload"])
            print(f"Status: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
