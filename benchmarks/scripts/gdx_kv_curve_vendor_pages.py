#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture vendor FA KV page bytes for the GDX bytes curve (fp8/fp4).

Product GDX fp8/fp4 deposits compress vendor FA storage only (uint8 e4m3 /
nvfp4 codes). There is no bf16→e4m3 cast on the serve path. This script runs a
short vLLM offline prefill with the same ``kv_cache_dtype`` as serve, then
depages FA pages into a numpy archive for ``gdx_kv_bytes_curve.py``.

Intended to run inside ``VLLM_IMAGE`` (or a host with matching vLLM), e.g.:

  docker run --rm --gpus all -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    -v MODEL:MODEL:ro -v OUT:OUT -v SCRIPT:SCRIPT:ro \\
    --entrypoint python3 VLLM_IMAGE SCRIPT \\
    --model-dir MODEL --kv-cache-dtype fp8 --max-t 8192 --out OUT/pages.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_text(path: Path | None) -> str:
    if path is not None and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    here = Path(__file__).resolve().parent
    for fixture in (
        here.parent / "fixtures" / "pride_and_prejudice.txt",
        here / "fixtures" / "pride_and_prejudice.txt",
    ):
        if fixture.is_file():
            return fixture.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(
        "real text required (--text-file or benchmarks/fixtures/pride_and_prejudice.txt)"
    )


def _staged_tensor(obj: Any) -> Any:
    """Unwrap StagedWriteTensor / UVA wrappers to a raw torch tensor."""
    import torch

    if torch.is_tensor(obj):
        return obj
    for attr in ("gpu", "tensor", "cpu", "uva"):
        inner = getattr(obj, attr, None)
        if inner is None:
            continue
        if torch.is_tensor(inner):
            return inner
        if hasattr(inner, "tensor") and torch.is_tensor(inner.tensor):
            return inner.tensor
    raise TypeError(f"cannot unwrap staged tensor from {type(obj)!r}")


def _num_computed(rs: Any, idx: int) -> int:
    import numpy as np

    nct = getattr(rs, "num_computed_tokens_np", None)
    if nct is not None:
        return int(np.asarray(nct).reshape(-1)[idx])
    nct = getattr(rs, "num_computed_tokens", None)
    if nct is None:
        return 0
    try:
        gpu = getattr(nct, "gpu", nct)
        return int(_staged_tensor(gpu)[idx].item())
    except Exception:
        return 0


def _depage_layer(
    kv: Any,
    block_ids: list[int],
    *,
    max_t: int,
) -> tuple[Any, Any]:
    """Depage FlashInfer packed FA fp8 ``[B,H,N,2D]`` uint8 → K/V ``[T,H,D]``."""
    import torch

    if not torch.is_tensor(kv):
        raise TypeError(f"kv_cache must be a tensor, got {type(kv)!r}")
    if kv.dtype != torch.uint8 and not str(kv.dtype).startswith("torch.float8"):
        raise RuntimeError(
            "gdx vendor pages fail-closed: expected FA fp8 uint8/float8 storage, "
            f"got dtype={kv.dtype}"
        )
    u8 = kv.view(torch.uint8) if kv.dtype != torch.uint8 else kv
    if int(u8.dim()) != 4:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: expected [B,H,N,2D] FA pages, got {tuple(u8.shape)}"
        )
    _b, h, block_size, c = (int(x) for x in u8.shape)
    if c % 2 != 0:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: packed last-dim not even (got {c})"
        )
    d = c // 2
    n_blocks_need = (int(max_t) + block_size - 1) // block_size
    if len(block_ids) < n_blocks_need:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: need {n_blocks_need} blocks, got {len(block_ids)}"
        )
    # Vectorized gather: token t → block_ids[t//bs], offset t%bs
    device = u8.device
    t_idx = torch.arange(int(max_t), device=device, dtype=torch.int64)
    blk = torch.tensor(block_ids[:n_blocks_need], device=device, dtype=torch.int64)
    page = blk[t_idx // block_size]
    off = t_idx % block_size
    # u8[page, :, off, :] → [T, H, 2D]
    gathered = u8[page, :, off, :]
    k = gathered[:, :, :d].contiguous()
    v = gathered[:, :, d:].contiguous()
    return k, v


def capture_vendor_pages(
    *,
    model_dir: Path,
    kv_cache_dtype: str,
    max_t: int,
    text_file: Path | None,
    gpu_memory_utilization: float = 0.90,
) -> dict[str, Any]:
    """Run one vLLM prefill and return numpy K/V stacks plus metadata."""
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    text = _load_text(text_file)
    max_t = int(max_t)
    if max_t < 1:
        raise ValueError("max_t must be positive")

    # +8: room for max_tokens=1 without exceeding context on chat templates.
    llm = LLM(
        model=str(model_dir),
        kv_cache_dtype=str(kv_cache_dtype),
        max_model_len=max_t + 8,
        enforce_eager=True,
        gpu_memory_utilization=float(gpu_memory_utilization),
        trust_remote_code=True,
        dtype="auto",
        enable_prefix_caching=False,
    )
    worker = llm.llm_engine.model_executor.driver_worker
    worker = getattr(worker, "worker", worker)
    runner = worker.model_runner
    if not getattr(runner, "kv_caches", None):
        raise RuntimeError("gdx vendor pages fail-closed: model_runner has no kv_caches")

    tok = llm.get_tokenizer()
    # ~4 chars/token headroom; then exact-trim to max_t ids.
    chunk = text[: max(max_t * 8, 50_000)]
    ids = tok.encode(chunk)
    if len(ids) < max_t:
        reps = (max_t // max(len(ids), 1)) + 2
        ids = (ids * reps)[:max_t]
    else:
        ids = ids[:max_t]
    prompt = tok.decode(ids)

    dump: dict[str, Any] = {"ok": False}

    orig = runner.execute_model

    def wrapped(scheduler_output, *args, **kwargs):  # noqa: ANN001
        out = orig(scheduler_output, *args, **kwargs)
        if dump.get("ok"):
            return out
        rs = runner.req_states
        if not getattr(rs, "req_id_to_index", None):
            return out
        rid, idx = next(iter(rs.req_id_to_index.items()))
        ncomp = _num_computed(rs, int(idx))
        if ncomp < max_t:
            return out
        bt0 = runner.block_tables.block_tables[0]
        tbl = _staged_tensor(bt0)
        row = tbl[int(idx)].detach().cpu().tolist()
        block_size = int(runner.kv_caches[0].shape[2])
        n_blocks = (max_t + block_size - 1) // block_size
        block_ids = [int(x) for x in row[:n_blocks]]
        if any(b < 0 for b in block_ids):
            raise RuntimeError("gdx vendor pages fail-closed: negative block id")
        ks: list[Any] = []
        vs: list[Any] = []
        with torch.inference_mode():
            for kv in runner.kv_caches:
                k, v = _depage_layer(kv, block_ids, max_t=max_t)
                ks.append(k.detach().cpu().numpy())
                vs.append(v.detach().cpu().numpy())
        k_all = np.stack(ks, axis=0)
        v_all = np.stack(vs, axis=0)
        dump.update(
            {
                "ok": True,
                "req_id": str(rid),
                "req_index": int(idx),
                "num_computed_tokens": int(ncomp),
                "block_ids": block_ids,
                "block_size": block_size,
                "k": k_all,
                "v": v_all,
                "kv_dtype": str(runner.kv_caches[0].dtype),
                "kv_shape0": tuple(int(x) for x in runner.kv_caches[0].shape),
            }
        )
        return out

    runner.execute_model = wrapped  # type: ignore[method-assign]
    try:
        llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))
    finally:
        runner.execute_model = orig  # type: ignore[method-assign]

    if not dump.get("ok"):
        raise RuntimeError(
            "gdx vendor pages fail-closed: prefill completed without FA page dump"
        )
    return dump


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--kv-cache-dtype",
        default=os.environ.get("SERVE_KV_CACHE_DTYPE") or "fp8",
        help="Must match serve (fp8 / nvfp4)",
    )
    parser.add_argument("--max-t", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args(argv)
    if not args.model_dir.is_dir():
        print(
            f"gdx_kv_curve_vendor_pages: FAIL model_dir missing: {args.model_dir}",
            file=sys.stderr,
        )
        return 2
    try:
        dump = capture_vendor_pages(
            model_dir=args.model_dir,
            kv_cache_dtype=str(args.kv_cache_dtype),
            max_t=int(args.max_t),
            text_file=args.text_file,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )
    except Exception as exc:  # noqa: BLE001 - meter must fail-closed
        print(
            f"gdx_kv_curve_vendor_pages: FAIL error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    import numpy as np

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "isiro-benchmark-gdx-vendor-pages-v1",
        "model_dir": args.model_dir.name,
        "kv_cache_dtype": str(args.kv_cache_dtype),
        "max_t": int(args.max_t),
        "layers": int(dump["k"].shape[0]),
        "kv_heads": int(dump["k"].shape[2]),
        "head_dim": int(dump["k"].shape[3]),
        "block_size": int(dump["block_size"]),
        "kv_dtype": dump["kv_dtype"],
        "kv_shape0": list(dump["kv_shape0"]),
        "block_ids": list(dump["block_ids"]),
        "kv_source": "vendor_fa_pages",
    }
    np.savez_compressed(
        args.out,
        k=dump["k"],
        v=dump["v"],
        meta_json=np.asarray([json.dumps(meta, sort_keys=True)]),
    )
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    if str(args.out).endswith(".npz"):
        meta_path = Path(str(args.out) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"gdx_kv_curve_vendor_pages: PASS layers={meta['layers']} "
        f"T={args.max_t} H={meta['kv_heads']} D={meta['head_dim']} "
        f"dtype={meta['kv_dtype']} out={args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
