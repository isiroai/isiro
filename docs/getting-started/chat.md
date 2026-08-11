---
title: Chat with your model
description: Connect a browser chat UI to your running isiro serve endpoint.
group: getting-started
order: 4
anchorPrefixes:
  - chat-ui
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

After [isiro serve](/docs/getting-started/run) is running, you can chat with the model through any client that speaks the OpenAI API at `http://<host>:8000/v1`. For example `curl`, an SDK, or a chat UI. One option is [Open WebUI](https://openwebui.com) (third-party); a Docker sample is below if you want to try it.

**Linux (Docker Engine)** sample. Host networking lets the container reach the API on `127.0.0.1:8000` directly:

```sh
docker run -d --network host \
  -e PORT=3000 \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 \
  -v open-webui:/app/backend/data \
  --name open-webui --restart always \
  ghcr.io/open-webui/open-webui:v0.9.6
```

**Docker Desktop (macOS / Windows)**: published port with the host-gateway alias:

```sh
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -v open-webui:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui --restart always \
  ghcr.io/open-webui/open-webui:v0.9.6
```

1. Open [http://localhost:3000](http://localhost:3000)
2. Create a local admin account
3. Select your model if it is not already selected, then chat

Wait for **ISIRO serve ready** before chatting. If you opened the UI earlier, refresh or reconnect once serve is up.

- `OPENAI_API_BASE_URL=…:8000/v1`: points Open WebUI at the serve API on first boot
- `--network host` (Linux): shares the host network so `127.0.0.1:8000` resolves; on Docker Desktop use `-p 3000:8080` + `--add-host=host.docker.internal:host-gateway` instead. With host networking, `-e PORT=3000` keeps the UI on port 3000 (Open WebUI defaults to 8080).
- `-v open-webui:…`: persist accounts and chat history across restarts
- `--restart always`: optional. Docker may bring the container back after reboot until you change or remove it

To stop the sample UI later:

```sh
docker stop open-webui
# optional: remove container and saved chat data
docker rm -f open-webui && docker volume rm open-webui
```

<DocsCollapsibleSection id="chat-ui-troubleshooting" title="Chat UI troubleshooting" expandForAnchors={['chat-ui-troubleshooting']}>

If the model list is empty or chat cannot reach the API, confirm serve is up (`ISIRO serve ready`) and that the UI can reach `:8000`.

On Linux, "no models" / a slow UI usually means the container cannot reach the serve API. A common cause is a host firewall dropping traffic from the Docker bridge to host ports, so the published-port + `host.docker.internal` pattern times out. Prefer the **Linux host-networking** sample above. Verify from inside the container:

```sh
docker exec open-webui curl -sf http://127.0.0.1:8000/v1/models
```

If that returns your model list, refresh [http://localhost:3000](http://localhost:3000). If you first tried the Docker Desktop command and it saved a bad connection, remove the container and volume (`docker rm -f open-webui && docker volume rm open-webui`) before re-running the host-networking command.

The sample pins Open WebUI `v0.9.6`. Newer 0.10.x builds can leave chat replies blank with some OpenAI-compatible backends; stick to the pinned tag unless you have a reason to upgrade.

</DocsCollapsibleSection>
