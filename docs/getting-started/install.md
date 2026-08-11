---
title: Installation
description: Install ISIRO compiler and runtime. Docker is required; NVIDIA GPU workloads also need the driver and Container Toolkit.
group: getting-started
order: 1
toc:
  - id: install-dependencies
    label: Dependencies
  - id: quick-install
    label: Quick install
  - id: inspect-before-running
    label: Inspect before running
  - id: maintenance
    label: Maintenance
anchorPrefixes:
  - install-dependencies
  - install-docker
  - install-nvidia-driver
  - platform-support
  - install-gpu
  - install-gpu-ubuntu
  - install-gpu-rhel
  - install-gpu-configure
  - install-gpu-verify
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

<DocsCollapsibleSection id="install-dependencies" title="Dependencies" expandForAnchors={['install-dependencies','install-docker','install-nvidia-driver','platform-support','install-gpu','install-gpu-ubuntu','install-gpu-rhel','install-gpu-configure','install-gpu-verify']} className="mb-10">

Docker is required on the host. Running models on an NVIDIA GPU also requires the NVIDIA driver and Container Toolkit on that host.

### 1. Docker

Check `docker version` to confirm setup. If it is not installed, install [Docker](https://docs.docker.com/get-docker/).

### 2. NVIDIA Driver

Confirm NVIDIA driver works on the host (`nvidia-smi` should succeed). If not, install [NVIDIA driver](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/introduction.html).

### 3. NVIDIA Container Toolkit

Required to use an NVIDIA GPU inside Docker. Check if you already have it installed ([Verify GPU Access in Docker](#install-gpu-verify)).

#### Install toolkit (Ubuntu / Debian)

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's|deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://|g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### Install toolkit (RHEL / Rocky / Amazon Linux)

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install -y nvidia-container-toolkit
```

#### Enable GPU Access in Docker

```sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Verify GPU Access in Docker

```sh
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

Any recent nvidia/cuda base image is fine.

For other distros, rootless Docker, or version-specific installs, see the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

</DocsCollapsibleSection>

## Quick install

```sh
curl -fsSL https://isiro.ai/install.sh | sh
```

By installing, you agree to [ISIRO EULA](/eula).

After installation, open a new terminal and run `isiro --help`.

## Inspect before running

If you prefer not to pipe the script directly, use the commands below to download, review, and verify its checksum before running it.

```sh
# Download
curl -fsSL https://isiro.ai/install.sh -o install.sh

# Review (optional)
less install.sh

# Verify checksum
curl -fsSL https://isiro.ai/install.sh.sha256 -o install.sh.sha256
sha256sum -c install.sh.sha256          # Linux
shasum -a 256 -c install.sh.sha256      # macOS

# Install
sh install.sh
```

## Maintenance

**Update:** re-run the install command. Upgrades in place; uninstall is not required.

**Uninstall:** `isiro uninstall`. See `isiro uninstall -h` for options.
