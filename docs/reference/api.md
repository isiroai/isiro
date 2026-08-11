---
title: API
description: OpenAI-compatible inference API served by isiro serve.
group: reference
order: 2
anchorPrefixes:
  - api-overview
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

`isiro serve` owns the inference API on port 8000. It is OpenAI-compatible, so you can use OpenAI clients and SDKs with it. Prometheus `/metrics` is on the same port.

Base URL: `http://HOST:8000/v1` (e.g. on localhost, `http://127.0.0.1:8000/v1`)

Smoke test:

```sh
curl http://127.0.0.1:8000/v1/models
```

