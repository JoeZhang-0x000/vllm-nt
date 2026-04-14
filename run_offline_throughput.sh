#!/usr/bin/env bash
set -euo pipefail

python "scripts/run_offline_throughput.py" run "$@"
