#!/usr/bin/env bash
set -euo pipefail

python "scripts/run_inference_validation.py" run "$@"
