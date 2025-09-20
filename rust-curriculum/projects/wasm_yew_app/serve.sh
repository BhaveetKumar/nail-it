#!/usr/bin/env bash
set -euo pipefail
wasm-pack build --target web
python3 -m http.server 8080