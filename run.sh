#!/usr/bin/env bash
# Linux/macOS launcher, for parity with run.ps1: starts the API in the background,
# waits for it to be ready, then runs Streamlit in the foreground. Thin deployment
# layer only — no application logic lives here.
#
# Usage: ./run.sh

set -euo pipefail

if [ ! -f .env ]; then
    echo "No .env file found. Copy .env.example to .env and fill in your provider credentials first." >&2
    exit 1
fi

echo "Starting the API (http://127.0.0.1:8000) ..."
uv run uvicorn app.main:app &
api_pid=$!
trap 'echo "Stopping the API ..."; kill "$api_pid" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 60); do
    if ! kill -0 "$api_pid" 2>/dev/null; then
        echo "The API process exited unexpectedly. Check the output above for the reason (e.g. missing credentials in .env)." >&2
        exit 1
    fi
    if curl --silent --fail --max-time 2 http://127.0.0.1:8000/documents > /dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != true ]; then
    echo "The API did not become ready within 60 seconds. Check the output above." >&2
    exit 1
fi

echo "API is ready. Starting the UI (http://127.0.0.1:8501) ..."
uv run streamlit run streamlit_app.py
