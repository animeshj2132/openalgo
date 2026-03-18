#!/usr/bin/env bash
set -e
cd /root/openalgo
# Prevent Flask-integrated websocket proxy from starting (we run openalgo-ws separately on 8765).
APP_MODE=standalone uv run gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:5000 app:app
