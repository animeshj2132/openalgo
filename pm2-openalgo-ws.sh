#!/usr/bin/env bash
set -e
cd /root/openalgo
uv run python -m websocket_proxy.server
