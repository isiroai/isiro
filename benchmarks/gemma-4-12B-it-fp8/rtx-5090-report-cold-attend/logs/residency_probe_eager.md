# Eager residency probe (follow-up)

`--enforce-eager`, `ISIRO_KV=bf16`, W=512, ~4504 prompt tokens.

| Event | Meter |
|---|---|
| Before EngineCore SIGTERM | `n_attach=96`, no deposit/skip fields |
| After atexit flush | `n_attach=96`, **`n_skip_unarmed=144`**, `attend_class=a`, `live_hbm=True` |

Conclusion: host hooks run under eager; cold deposit is skipped as unarmed (`persist_armed=false`), not because CUSTOM FA never enters the hook.
