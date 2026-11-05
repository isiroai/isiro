# Scope: FA cold-attend remeter (not GLM MLA)

This stamp is **Gemma FA `path_a` / `fa-kv`** on `rtx-5090` (`ISIRO_KV=bf16`).

It shares the o1 cold/hot spine with GLM, but it is **not** a proxy for
`path_a_mla` / `o1-mla-latent`. Do not claim these numbers unlock GLM on
2x RTX PRO 6000 or prove MLA vendor-page reclaim.

Dual GO for `report.md`:

1. Physical cold-attend / residency after vendor Sliding or pool reclaim
2. Generation latency beats the published `rtx-5090-report` baseline table
