#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/tmp/ninetoothed-examples}"

if [ -d "$TARGET_DIR/.git" ]; then
  git -C "$TARGET_DIR" pull --ff-only
else
  git clone https://github.com/InfiniTensor/ninetoothed-examples.git "$TARGET_DIR"
fi
