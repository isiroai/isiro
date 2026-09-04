#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture or compare greedy serve token IDs (baseline vs TIC).

Runs against an already-up OpenAI-compatible server (vLLM / isiro serve).
Capture writes per-variant JSON; compare folds both sides into a match report.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "isiro-benchmark-output-match-v1"
THINK_MARKERS = ("<think>", "</think>", "<|im_start|>think")

# Fixed prompts: short, diverse enough to exercise chat + greedy decode.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Reply with exactly one word: ping",
    "What is 2+2? Answer with only the number.",
    "Name the capital of France in one word.",
    "Reply with exactly one word in lowercase: hello",
)


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_token_ids(choice: dict[str, Any]) -> list[int] | None:
    for key in ("token_ids", "output_token_ids"):
        value = choice.get(key)
        if isinstance(value, list) and value and all(isinstance(x, int) for x in value):
            return [int(x) for x in value]
    message = choice.get("message") or {}
    if isinstance(message, dict):
        for key in ("token_ids", "output_token_ids"):
            value = message.get(key)
            if (
                isinstance(value, list)
                and value
                and all(isinstance(x, int) for x in value)
            ):
                return [int(x) for x in value]
    # logprobs token-id form: "token": "id:1234" (vLLM return_tokens_as_token_ids)
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if isinstance(content, list) and content:
        ids: list[int] = []
        for item in content:
            if not isinstance(item, dict):
                return None
            token = item.get("token")
            if isinstance(token, str) and token.startswith("token_id:"):
                try:
                    ids.append(int(token.split(":", 1)[1]))
                    continue
                except ValueError:
                    return None
            if isinstance(token, str) and token.startswith("id:"):
                try:
                    ids.append(int(token.split(":", 1)[1]))
                    continue
                except ValueError:
                    return None
            return None
        return ids or None
    return None


def _extract_text(choice: dict[str, Any]) -> str:
    message = choice.get("message") or {}
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _extract_finish_reason(choice: dict[str, Any]) -> str | None:
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else None


def thinking_controls_present(payload: dict[str, Any]) -> bool:
    """True when the request disables thinking (kwargs and/or /no_think)."""
    kwargs = payload.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False:
        return True
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and "/no_think" in content:
            return True
    return False


def _with_no_think_message(payload: dict[str, Any]) -> dict[str, Any]:
    retry = dict(payload)
    messages: list[dict[str, Any]] = []
    for message in retry.get("messages") or []:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        content = item.get("content")
        if (
            item.get("role") == "user"
            and isinstance(content, str)
            and "/no_think" not in content
        ):
            item["content"] = content.rstrip() + "\n/no_think"
        messages.append(item)
    retry["messages"] = messages
    return retry


def greedy_chat_payload(
    *,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    request_token_ids: bool = True,
) -> dict[str, Any]:
    """temp=0 chat/completions body. Always disable thinking for greedy IDs."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "logprobs": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if request_token_ids:
        payload["return_token_ids"] = True
        payload["return_tokens_as_token_ids"] = True
    return payload


def _post_greedy(
    chat_url: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Retry 400s without dropping thinking-off. Token-id flags may be stripped."""
    if not thinking_controls_present(payload):
        raise RuntimeError("greedy payload is missing thinking-off controls")
    try:
        return _post_json(chat_url, payload, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code != 400:
            raise RuntimeError(
                f"chat/completions failed ({exc.code}): {body[:400]}"
            ) from exc
        lowered = body.lower()
        retry = dict(payload)
        if "return_token" in lowered:
            retry.pop("return_token_ids", None)
            retry.pop("return_tokens_as_token_ids", None)
            if not thinking_controls_present(retry):
                raise RuntimeError(
                    "refusing token-id retry without thinking-off controls"
                ) from exc
            try:
                return _post_json(chat_url, retry, timeout=timeout)
            except urllib.error.HTTPError as retry_exc:
                body = retry_exc.read().decode("utf-8", errors="replace")
                if retry_exc.code != 400:
                    raise RuntimeError(
                        f"chat/completions failed ({retry_exc.code}): {body[:400]}"
                    ) from retry_exc
                lowered = body.lower()
                retry = dict(retry)
        if "chat_template" in lowered or "enable_thinking" in lowered:
            retry = _with_no_think_message(retry)
            retry.pop("chat_template_kwargs", None)
            if not thinking_controls_present(retry):
                raise RuntimeError(
                    "refusing chat_template retry without /no_think"
                ) from exc
            return _post_json(chat_url, retry, timeout=timeout)
        raise RuntimeError(
            f"chat/completions failed ({exc.code}): {body[:400]}"
        ) from exc


def greedy_protocol_errors(capture: dict[str, Any]) -> list[str]:
    """Thinking must be off. Token-ID compare does not depend on max_tokens."""
    errors: list[str] = []
    if capture.get("enable_thinking") is not False:
        errors.append("enable_thinking must be false on greedy captures")
    for item in capture.get("prompts") or []:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").lower()
        if any(marker in text for marker in THINK_MARKERS):
            errors.append(
                f"prompt {item.get('index')}: thinking markers in text"
            )
    return errors


def capture_variant(
    *,
    api_url: str,
    model: str,
    variant: str,
    seed: int,
    max_tokens: int,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    timeout: float = 120.0,
) -> dict[str, Any]:
    base = api_url.rstrip("/")
    if not base.endswith("/v1"):
        chat_url = f"{base}/v1/chat/completions"
    else:
        chat_url = f"{base}/chat/completions"
    results: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        payload = greedy_chat_payload(
            model=model,
            prompt=prompt,
            seed=seed,
            max_tokens=max_tokens,
        )
        response = _post_greedy(chat_url, payload, timeout)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"empty choices for prompt index {index}")
        choice = choices[0]
        token_ids = _extract_token_ids(choice)
        text = _extract_text(choice)
        finish_reason = _extract_finish_reason(choice)
        if token_ids is None and not text:
            raise RuntimeError(
                f"no token_ids or text for prompt index {index}; "
                f"choice keys={sorted(choice)}"
            )
        results.append(
            {
                "index": index,
                "prompt": prompt,
                "token_ids": token_ids,
                "text": text,
                "finish_reason": finish_reason,
            }
        )
    doc = {
        "schema": SCHEMA,
        "kind": "capture",
        "variant": variant,
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt_count": len(results),
        "enable_thinking": False,
        "prompts": results,
    }
    protocol = greedy_protocol_errors(doc)
    if protocol:
        raise RuntimeError(
            "greedy capture failed protocol (thinking still on): "
            + "; ".join(protocol)
        )
    return doc


def compare_captures(
    baseline: dict[str, Any],
    tic: dict[str, Any],
) -> dict[str, Any]:
    b_prompts = baseline.get("prompts") or []
    t_prompts = tic.get("prompts") or []
    matched = 0
    details: list[dict[str, Any]] = []
    n = min(len(b_prompts), len(t_prompts))
    for i in range(n):
        b = b_prompts[i]
        t = t_prompts[i]
        b_ids = b.get("token_ids")
        t_ids = t.get("token_ids")
        if isinstance(b_ids, list) and isinstance(t_ids, list):
            ok = b_ids == t_ids
            mode = "token_ids"
        else:
            ok = (b.get("text") or "") == (t.get("text") or "")
            mode = "text_fallback"
        if ok:
            matched += 1
        details.append(
            {
                "index": i,
                "ok": ok,
                "compare_mode": mode,
                "baseline_token_ids": b_ids,
                "tic_token_ids": t_ids,
                "baseline_text": b.get("text"),
                "tic_text": t.get("text"),
            }
        )
    protocol_errors = greedy_protocol_errors(baseline) + greedy_protocol_errors(tic)
    protocol_ok = not protocol_errors
    all_ok = (
        protocol_ok
        and len(b_prompts) == len(t_prompts)
        and len(b_prompts) > 0
        and matched == len(b_prompts)
        and all(item["ok"] for item in details)
    )
    return {
        "schema": SCHEMA,
        "kind": "compare",
        "prompt_count": len(b_prompts),
        "matched": matched,
        "serve_output_match_ok": all_ok,
        "protocol_ok": protocol_ok,
        "protocol_errors": protocol_errors,
        "details": details,
        "baseline_seed": baseline.get("seed"),
        "tic_seed": tic.get("seed"),
        "max_tokens": baseline.get("max_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Capture greedy outputs from a live server")
    capture.add_argument("--api-url", required=True)
    capture.add_argument("--model", required=True)
    capture.add_argument("--variant", required=True, choices=("baseline", "isiro"))
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--seed", type=int, default=0)
    capture.add_argument("--max-tokens", type=int, default=32)

    compare = sub.add_parser(
        "compare", help="Compare baseline/isiro capture JSON files"
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--tic", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "capture":
        doc = capture_variant(
            api_url=args.api_url,
            model=args.model,
            variant=args.variant,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.out)
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    tic = json.loads(args.tic.read_text(encoding="utf-8"))
    doc = compare_captures(baseline, tic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0 if doc.get("serve_output_match_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
