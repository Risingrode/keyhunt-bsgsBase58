#!/bin/bash
# Regression test: completed runs should report a wall-clock runtime even
# when periodic stats are disabled.

set -euo pipefail

if [ ! -x ./keyhunt ]; then
    echo "[ERROR] keyhunt binary not found; run make first" >&2
    exit 1
fi

RESULT=$(timeout 20 ./keyhunt -m address -f tests/66.txt -r 1:400 -l compress -q -s 0 -n 1024 2>&1)

if ! grep -Eq "\\[\\+\\] Elapsed time: [0-9]{2}:[0-9]{2}:[0-9]{2}" <<< "$RESULT"; then
    echo "$RESULT"
    echo "[FAIL] Final elapsed runtime was not reported" >&2
    exit 1
fi

echo "[PASS] Final elapsed runtime was reported"
