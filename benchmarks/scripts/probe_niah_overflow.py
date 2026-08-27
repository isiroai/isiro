#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated NIAH overflow probe: in-pool vs past-pool on a running TIC serve.

Does not run the heatmap. Expects isiro serve at --url with product kv pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from needle_haystack import _one_probe  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--kv-gib", type=float, default=0.25)
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()
    rows = []
    for ctx in (4000, 8000, 16000, 32000):
        for depth in (0.0, 100.0):
            row = _one_probe(
                url=args.url,
                model=args.model,
                panel="tic",
                kv_cache_gib=float(args.kv_gib),
                depth=depth,
                context_tokens=ctx,
                timeout=float(args.timeout),
            )
            rows.append(row)
            print(
                f"overflow_probe T={ctx} depth={depth:.0f} "
                f"completes={row.get('response_completes')} "
                f"hit={row.get('hit')} oom={row.get('oom')} "
                f"err={row.get('error', '')!s:.180}",
                flush=True,
            )
    print(json.dumps({"rows": rows}, sort_keys=True))
    in_pool = [r for r in rows if int(r["context_tokens"]) == 4000]
    past = [r for r in rows if int(r["context_tokens"]) > 4000]
    ok_in = all(r.get("response_completes") and r.get("hit") for r in in_pool)
    ok_past = all(r.get("response_completes") and r.get("hit") for r in past)
    if not ok_in:
        return 2
    return 0 if ok_past else 1


if __name__ == "__main__":
    raise SystemExit(main())
