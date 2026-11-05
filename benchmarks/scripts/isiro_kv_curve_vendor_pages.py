#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture vendor FA KV page bytes for the Isiro KV bytes curve (fp8/fp4).

Product Isiro KV fp8/fp4 deposits compress vendor FA storage only (uint8 e4m3 /
nvfp4 codes). There is no bf16→e4m3 cast on the serve path. This script runs a
short vLLM offline prefill with the same ``kv_cache_dtype`` as serve, then
depages FA pages into a numpy archive for ``isiro_kv_bytes_curve.py``.

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


def _load_hf_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def _is_packed_fa_layer_type(name: str) -> bool:
    """True for dense FA / sliding FA; false for KDA/Mamba/MLA/sparse."""
    s = str(name).lower()
    if "linear" in s or "mamba" in s or s in ("gdn", "ssm", "kda"):
        return False
    if "sparse" in s or "mla" in s:
        return False
    if "full" in s or "sliding" in s or s in ("attention", "full_attention"):
        return True
    return False


def vendor_fa_pages_unsupported_reason(model_dir: Path) -> str | None:
    """Why path-A packed-FA page dump cannot meter this HF config.

    GLM-5.3-Flash / DeepSeek-V4.1-Flash use MLA / sparse / CSA2 / KDA pages
    (``MLAAttentionSpec``), not Llama-style packed FA ``[B,H,N,2D]`` uint8.
    Qwen3.5 hybrid still has ``full_attention`` FA pages and is allowed.
    """
    cfg = _load_hf_config(model_dir)
    if not cfg:
        return None
    maps = [cfg]
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        maps.append(tc)
    # Product serve: ISIRO_KV auto-routes these to unique_c (see gdx_route.py).
    # This script is the path_a FA-page curve only.
    _unique_c_hint = (
        "serve ISIRO_KV auto-routes unique_c; this path_a FA-page curve does not apply"
    )
    for m in maps:
        if m.get("kv_lora_rank") is not None:
            return (
                "HF kv_lora_rank set: MLA latent pages, not packed FA GQA "
                f"({_unique_c_hint})"
            )
        if m.get("kv_source_layer_ids"):
            return (
                "HF kv_source_layer_ids set: CSA2 shared-source KV, not per-layer "
                f"packed FA ({_unique_c_hint})"
            )
        if m.get("sparse_attention_enabled") or m.get("sparse_attention_config"):
            return (
                "HF sparse_attention_* set: sparse MLA pages, not packed FA "
                f"({_unique_c_hint})"
            )
    layer_types = None
    for m in maps:
        lt = m.get("layer_types")
        if isinstance(lt, list) and lt:
            layer_types = lt
            break
    if layer_types is not None:
        packed = [lt for lt in layer_types if _is_packed_fa_layer_type(str(lt))]
        if not packed:
            sample = sorted({str(x) for x in layer_types})[:6]
            return (
                "HF layer_types has no packed FA layers "
                f"(saw {sample}); only linear/KDA/sparse/MLA. "
                f"{_unique_c_hint}"
            )
    return None


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
    """Live computed-token count (prefer staged GPU tensor; np mirror is often stale)."""
    nct = getattr(rs, "num_computed_tokens", None)
    if nct is not None:
        try:
            return int(_staged_tensor(nct)[idx].item())
        except Exception:
            pass
    nct_np = getattr(rs, "num_computed_tokens_np", None)
    if nct_np is not None:
        import numpy as np

        try:
            return int(np.asarray(nct_np).reshape(-1)[idx])
        except Exception:
            return 0
    return 0


def _parse_fa_shape(shape: tuple[int, ...]) -> tuple[str, int, int, int, int]:
    """Return ``(layout, blocks, heads, block_size, packed_c)``.

    FlashInfer may use HND ``[B,H,N,C]`` or NHD/LBNHC ``[B,N,H,C]``. Hybrid
    Qwen often pads FA block_size to match Mamba (e.g. N=1568).
    """
    if len(shape) != 4:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: expected 4D FA pages, got {shape}"
        )
    b, d1, d2, c = (int(x) for x in shape)
    if c < 2 or c % 2 != 0:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: packed last-dim not even (got {c})"
        )
    # Heads are small; block_size is the large page extent (16…few thousand).
    if d1 <= 256 and d2 >= max(d1, 8):
        return "hnd", b, d1, d2, c
    if d2 <= 256 and d1 >= max(d2, 8):
        return "nhd", b, d2, d1, c
    # Default HND (dense Llama-style FA).
    return "hnd", b, d1, d2, c


def _fa_kv_caches(runner: Any) -> list[Any]:
    """Select FlashAttention-style packed FA pages; skip hybrid Mamba/GDN slabs."""
    import torch

    out: list[Any] = []
    for kv in getattr(runner, "kv_caches", None) or []:
        if not torch.is_tensor(kv):
            continue
        if kv.dtype != torch.uint8 and not str(kv.dtype).startswith("torch.float8"):
            continue
        if int(kv.dim()) != 4:
            continue
        try:
            _layout, _b, h, block_size, c = _parse_fa_shape(tuple(int(x) for x in kv.shape))
        except RuntimeError:
            continue
        if h < 1 or block_size < 1 or c < 2:
            continue
        out.append(kv)
    return out


def _manager_block_size(runner: Any, storage_block_size: int) -> int:
    """Allocator / hybrid page size (may exceed FlashInfer storage tile)."""
    candidates: list[Any] = []
    cc = getattr(runner, "cache_config", None)
    if cc is not None:
        candidates.append(getattr(cc, "block_size", None))
    vcfg = getattr(runner, "vllm_config", None)
    if vcfg is not None:
        vcc = getattr(vcfg, "cache_config", None)
        if vcc is not None:
            candidates.append(getattr(vcc, "block_size", None))
        candidates.append(getattr(vcfg, "attention_block_size", None))
    best = int(storage_block_size)
    for val in candidates:
        try:
            iv = int(val) if val is not None else 0
        except (TypeError, ValueError):
            continue
        if iv >= best:
            best = iv
    return best


def _expand_logical_block_ids(
    logical_ids: list[int],
    *,
    manager_block_size: int,
    storage_block_size: int,
    n_storage_blocks: int,
    n_logical_blocks: int,
) -> list[int]:
    """Map hybrid manager pages (e.g. 1568) onto FlashInfer storage tiles (e.g. 32)."""
    mgr = int(manager_block_size)
    stor = int(storage_block_size)
    ids = [int(x) for x in logical_ids]
    if mgr <= stor or mgr % stor != 0:
        return ids[:n_storage_blocks]
    tiles = mgr // stor
    max_id = max(ids) if ids else 0
    # Small ids are manager page indices; large aligned ids are storage starts.
    index_mode = max_id < max(int(n_logical_blocks) * 4, 8)
    out: list[int] = []
    for lid in ids:
        base = int(lid) * tiles if index_mode else int(lid)
        out.extend(range(base, base + tiles))
        if len(out) >= n_storage_blocks:
            break
    if len(out) < n_storage_blocks:
        start = out[-1] + 1 if out else 0
        out.extend(range(start, start + (n_storage_blocks - len(out))))
    return out[:n_storage_blocks]


def _read_block_table_rows(runner: Any, idx: int) -> list[list[int]]:
    """All block-table rows for req index (hybrid models have multiple groups)."""
    rows: list[list[int]] = []
    bt_root = getattr(runner, "block_tables", None)
    tables = []
    if bt_root is not None:
        tables = list(getattr(bt_root, "block_tables", None) or [])
        if not tables and hasattr(bt_root, "__iter__"):
            try:
                tables = list(bt_root)
            except Exception:
                tables = []
    for bt in tables:
        try:
            row = _staged_tensor(bt)[int(idx)].detach().cpu().tolist()
        except Exception:
            continue
        rows.append([int(x) for x in row])
    return rows


def _block_ids_for_fa(
    runner: Any,
    *,
    idx: int,
    max_t: int,
    block_size: int,
) -> tuple[list[int], str]:
    """Resolve storage-tile FA block ids. Prefer live tables over sequential guess."""
    storage_bs = int(block_size)
    n_storage = (int(max_t) + storage_bs - 1) // storage_bs
    mgr_bs = _manager_block_size(runner, storage_bs)
    n_logical = (int(max_t) + int(mgr_bs) - 1) // int(mgr_bs)
    rows = _read_block_table_rows(runner, idx=int(idx))
    best: list[int] | None = None
    best_score = -1
    # Prefer the group with the most distinct non-negative ids in the logical span.
    for row in reversed(rows):
        logical = [int(x) for x in row[:n_logical]]
        if not logical or any(b < 0 for b in logical):
            continue
        # Allow leading 0 (valid block id); require enough slots for max_t.
        if len(logical) < n_logical:
            continue
        nonzeroish = sum(1 for b in logical if b != 0)
        uniq = len({b for b in logical if b >= 0})
        # Score: prefer more unique pages; tolerate all-zero only if single page.
        score = uniq * 1000 + nonzeroish
        if n_logical == 1 and logical[0] >= 0:
            score = max(score, 1)
        if score > best_score and (uniq > 1 or logical[0] == 0 or nonzeroish > 0):
            best_score = score
            best = logical
    if best is not None and best_score > 0:
        expanded = _expand_logical_block_ids(
            best,
            manager_block_size=mgr_bs,
            storage_block_size=storage_bs,
            n_storage_blocks=n_storage,
            n_logical_blocks=n_logical,
        )
        return expanded, f"table_mgr{mgr_bs}_stor{storage_bs}"
    # Prefix-cache off + one request: FA pages fill contiguous storage tiles.
    return list(range(n_storage)), "sequential"


def _depage_layer(
    kv: Any,
    block_ids: list[int],
    *,
    max_t: int,
) -> tuple[Any, Any]:
    """Depage FlashInfer packed FA fp8 pages → K/V ``[T,H,D]``."""
    import torch

    if not torch.is_tensor(kv):
        raise TypeError(f"kv_cache must be a tensor, got {type(kv)!r}")
    if kv.dtype != torch.uint8 and not str(kv.dtype).startswith("torch.float8"):
        raise RuntimeError(
            "gdx vendor pages fail-closed: expected FA fp8 uint8/float8 storage, "
            f"got dtype={kv.dtype}"
        )
    u8 = kv.view(torch.uint8) if kv.dtype != torch.uint8 else kv
    layout, _b, h, block_size, c = _parse_fa_shape(tuple(int(x) for x in u8.shape))
    d = c // 2
    n_blocks_need = (int(max_t) + block_size - 1) // block_size
    if len(block_ids) < n_blocks_need:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: need {n_blocks_need} blocks, got {len(block_ids)}"
        )
    device = u8.device
    t_idx = torch.arange(int(max_t), device=device, dtype=torch.int64)
    blk = torch.tensor(block_ids[:n_blocks_need], device=device, dtype=torch.int64)
    page = blk[t_idx // block_size]
    off = t_idx % block_size
    if layout == "hnd":
        gathered = u8[page, :, off, :]
    else:
        gathered = u8[page, off, :, :]
    if int(gathered.shape[1]) != h:
        raise RuntimeError(
            f"gdx vendor pages fail-closed: layout={layout} gather H mismatch "
            f"{tuple(gathered.shape)} vs H={h}"
        )
    k = gathered[:, :, :d].contiguous()
    v = gathered[:, :, d:].contiguous()
    return k, v


def _coverage_tokens(k_all: Any, *, max_t: int) -> int:
    """Count leading token positions with any non-zero FA code."""
    import numpy as np

    k = np.asarray(k_all)
    if k.ndim != 4:
        return 0
    t_max = min(int(max_t), int(k.shape[1]))
    nz = (k[:, :t_max] != 0).any(axis=(0, 2, 3))
    if not bool(nz.any()):
        return 0
    # Allow sparse holes; count last True index + 1 as covered span.
    return int(np.where(nz)[0][-1]) + 1


def _materialize_dump(
    *,
    runner: Any,
    fa_caches: list[Any],
    max_t: int,
    idx: int,
    rid: str,
    ncomp: int,
    block_ids: list[int] | None = None,
    block_ids_source: str = "",
) -> dict[str, Any]:
    import numpy as np
    import torch

    layout, _b, _h, block_size, _c = _parse_fa_shape(
        tuple(int(x) for x in fa_caches[0].shape)
    )
    source = str(block_ids_source or "")
    if block_ids is None:
        block_ids, source = _block_ids_for_fa(
            runner, idx=int(idx), max_t=max_t, block_size=block_size
        )
    if any(b < 0 for b in block_ids):
        raise RuntimeError("gdx vendor pages fail-closed: negative block id")
    ks: list[Any] = []
    vs: list[Any] = []
    with torch.inference_mode():
        for kv in fa_caches:
            k, v = _depage_layer(kv, block_ids, max_t=max_t)
            ks.append(k.detach().cpu().numpy())
            vs.append(v.detach().cpu().numpy())
    k_all = np.stack(ks, axis=0)
    v_all = np.stack(vs, axis=0)
    covered = _coverage_tokens(k_all, max_t=max_t)
    # Hybrid wrong-table gather often fills only one manager page (~1568).
    min_cover = max(int(max_t) * 9 // 10, int(max_t) - int(block_size))
    if covered < min_cover:
        raise RuntimeError(
            "gdx vendor pages fail-closed: FA dump coverage too low "
            f"(covered_t={covered} max_t={max_t} min={min_cover} "
            f"source={source} n_blocks={len(block_ids)} "
            f"uniq_blocks={len(set(block_ids))})"
        )
    return {
        "ok": True,
        "req_id": str(rid),
        "req_index": int(idx),
        "num_computed_tokens": int(ncomp),
        "block_ids": list(block_ids),
        "block_ids_source": source,
        "covered_t": int(covered),
        "block_size": int(block_size),
        "layout": layout,
        "k": k_all,
        "v": v_all,
        "kv_dtype": str(fa_caches[0].dtype),
        "kv_shape0": tuple(int(x) for x in fa_caches[0].shape),
        "n_fa_layers": int(len(fa_caches)),
        "n_cache_tensors": int(len(getattr(runner, "kv_caches", []) or [])),
    }


def capture_vendor_pages(
    *,
    model_dir: Path,
    kv_cache_dtype: str,
    max_t: int,
    text_file: Path | None,
    gpu_memory_utilization: float = 0.90,
) -> dict[str, Any]:
    """Run one vLLM prefill and return numpy K/V stacks plus metadata."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    reason = vendor_fa_pages_unsupported_reason(Path(model_dir))
    if reason:
        raise RuntimeError(f"gdx vendor pages fail-closed: {reason}")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    text = _load_text(text_file)
    max_t = int(max_t)
    if max_t < 1:
        raise ValueError("max_t must be positive")

    # +8: room for max_tokens=1.
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
    fa_caches = _fa_kv_caches(runner)
    if not fa_caches:
        raise RuntimeError(
            "gdx vendor pages fail-closed: no FA uint8 packed kv_caches "
            f"(saw {len(getattr(runner, 'kv_caches', []) or [])} cache tensors)"
        )

    tok = llm.get_tokenizer()
    # Exact max_t token ids; TokensPrompt avoids chat-template re-encode drift.
    chunk = text[: max(max_t * 8, 50_000)]
    ids = list(tok.encode(chunk))
    if not ids:
        raise RuntimeError("gdx vendor pages fail-closed: tokenizer produced no ids")
    while len(ids) < max_t:
        ids.extend(ids)
    ids = ids[:max_t]

    dump: dict[str, Any] = {"ok": False, "last_ncomp": 0}
    stashed_ids: list[int] | None = None
    stashed_source = ""
    stashed_idx = 0
    stashed_rid = ""
    layout0, _b0, _h0, storage_bs, _c0 = _parse_fa_shape(
        tuple(int(x) for x in fa_caches[0].shape)
    )
    del layout0, _b0, _h0, _c0

    orig = runner.execute_model

    def wrapped(scheduler_output, *args, **kwargs):  # noqa: ANN001
        nonlocal stashed_ids, stashed_source, stashed_idx, stashed_rid
        out = orig(scheduler_output, *args, **kwargs)
        if dump.get("ok"):
            return out
        rs = runner.req_states
        if not getattr(rs, "req_id_to_index", None):
            return out
        rid, idx = next(iter(rs.req_id_to_index.items()))
        ncomp = _num_computed(rs, int(idx))
        dump["last_ncomp"] = int(ncomp)
        # Stash live tables while the request still owns pages (hybrid clears
        # after generate; sequential fallback then only sees one manager page).
        ids, source = _block_ids_for_fa(
            runner, idx=int(idx), max_t=max_t, block_size=storage_bs
        )
        if source.startswith("table"):
            stashed_ids = list(ids)
            stashed_source = source
            stashed_idx = int(idx)
            stashed_rid = str(rid)
        # Prefill may report ncomp == len(ids) before the decode step.
        if ncomp < max_t:
            return out
        dump.update(
            _materialize_dump(
                runner=runner,
                fa_caches=fa_caches,
                max_t=max_t,
                idx=int(idx),
                rid=str(rid),
                ncomp=int(ncomp),
                block_ids=stashed_ids,
                block_ids_source=stashed_source or source,
            )
        )
        return out

    runner.execute_model = wrapped  # type: ignore[method-assign]
    try:
        llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        # Hybrid / staged counters sometimes never surface ncomp>=max_t in the
        # wrap; reuse stashed table ids (never blind sequential after clear).
        if not dump.get("ok"):
            if stashed_ids is None:
                raise RuntimeError(
                    "gdx vendor pages fail-closed: no live FA block table during "
                    f"prefill (last_ncomp={dump.get('last_ncomp')} max_t={max_t})"
                )
            dump.update(
                _materialize_dump(
                    runner=runner,
                    fa_caches=fa_caches,
                    max_t=max_t,
                    idx=int(stashed_idx),
                    rid=str(stashed_rid or "post_generate"),
                    ncomp=int(dump.get("last_ncomp") or max_t),
                    block_ids=stashed_ids,
                    block_ids_source=stashed_source,
                )
            )
    finally:
        runner.execute_model = orig  # type: ignore[method-assign]

    if not dump.get("ok"):
        raise RuntimeError(
            "gdx vendor pages fail-closed: prefill completed without FA page dump "
            f"(last_ncomp={dump.get('last_ncomp')} max_t={max_t} "
            f"n_fa={len(fa_caches)})"
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
            f"isiro_kv_curve_vendor_pages: FAIL model_dir missing: {args.model_dir}",
            file=sys.stderr,
        )
        return 2
    reason = vendor_fa_pages_unsupported_reason(args.model_dir)
    if reason:
        print(f"isiro_kv_curve_vendor_pages: FAIL {reason}", file=sys.stderr)
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
            f"isiro_kv_curve_vendor_pages: FAIL error={type(exc).__name__}: {exc}",
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
        "block_ids_source": dump.get("block_ids_source"),
        "covered_t": dump.get("covered_t"),
        "layout": dump.get("layout"),
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
        f"isiro_kv_curve_vendor_pages: PASS layers={meta['layers']} "
        f"T={args.max_t} H={meta['kv_heads']} D={meta['head_dim']} "
        f"dtype={meta['kv_dtype']} out={args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
