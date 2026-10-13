#!/usr/bin/env python3
"""Summarize residency scrape into physical vs ledger fields for report.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _meter(blob: dict[str, Any]) -> dict[str, Any]:
    for key in ("meter_after_long", "meter_after_short"):
        m = blob.get(key)
        if isinstance(m, dict) and m:
            return m
    return {}


def summarize(scrape: dict[str, Any]) -> dict[str, Any]:
    m = _meter(scrape)
    dual = m.get("dual_hold")
    if dual is None:
        dual = scrape.get("dual_hold")
    live_fa = m.get("live_fa_nbytes", scrape.get("live_fa_nbytes"))
    freed = m.get("freed_fa_nbytes", scrape.get("freed_fa_nbytes"))
    save = m.get("save_vs_all_fa", scrape.get("save_vs_all_fa"))
    f_cold = m.get("f_cold", scrape.get("f_cold"))
    n_persist = m.get("n_live_persist", scrape.get("n_live_persist"))
    n_deposit = m.get("n_deposit", scrape.get("n_deposit"))
    attend = m.get("attend_class", scrape.get("attend_class"))

    # Ledger twin free: free_fa_twin accounting (not nvidia-smi by itself).
    ledger_twin_free = bool(dual is False or dual == 0) and (
        freed is not None and int(freed or 0) > 0
    )
    # Physical reclaim signal for FA path_a: vendor Sliding/pool must have
    # run (persist>0, f_cold>0) and live twin ledger cleared after deposit.
    physical_reclaim_armed = bool(
        ledger_twin_free
        and float(f_cold or 0) > 0
        and int(n_persist or n_deposit or 0) > 0
        and int(live_fa or 0) == 0
    )

    niah = {
        k: scrape[k]
        for k in scrape
        if k.startswith("niah_") and not k.endswith(("_ms", "_err", "_skip"))
    }
    niah_ok = all(v is None or float(v) >= 1.0 for v in niah.values()) if niah else False

    return {
        "route": "path_a",
        "payload": "fa-kv",
        "not_mla_proxy": True,
        "dual_hold": dual,
        "f_cold": f_cold,
        "save_vs_all_fa": save,
        "live_fa_nbytes": live_fa,
        "freed_fa_nbytes": freed,
        "n_live_persist": n_persist,
        "n_deposit": n_deposit,
        "attend_class": attend,
        "gpu_kv_tokens": scrape.get("gpu_kv_tokens"),
        "long_ok": scrape.get("long_ok"),
        "ledger_twin_free": ledger_twin_free,
        "physical_reclaim_armed": physical_reclaim_armed,
        "niah": niah,
        "niah_all_pass": niah_ok,
        "residency_go": bool(
            physical_reclaim_armed and scrape.get("long_ok") and niah_ok
        ),
        "note": (
            "physical_reclaim_armed means cold deposit past W with free_fa_twin "
            "ledger clear (live_fa_nbytes=0) after vendor Sliding/pool-armed "
            "persist. Reserved KV pool tokens remain vendor admission slots."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scrape = json.loads(args.scrape.read_text(encoding="utf-8"))
    if "gdx" in scrape and isinstance(scrape["gdx"], dict):
        scrape = scrape["gdx"]
    out = summarize(scrape)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
