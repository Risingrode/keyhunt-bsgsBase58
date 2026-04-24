#!/bin/bash
# Contract test: a non-CUDA build must not silently run WIF recovery on CPU
# when the user explicitly requests -g.

set -euo pipefail

if [ ! -x ./keyhunt ]; then
    echo "[ERROR] keyhunt binary not found; run make first" >&2
    exit 1
fi

PARTIAL_WIF="KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFf*EZnLdvkn"
PUBKEY="02e963ffdfe34e63b68aeb42a5826e08af087660e0dac1c3e79f7625ca4e6ae482"

set +e
RESULT=$(timeout 60 ./keyhunt -m wif-recovery -p "$PARTIAL_WIF" -P "$PUBKEY" -n 0x100000 -q -s 0 -g 2>&1)
STATUS=$?
set -e

if [ "$STATUS" -eq 0 ]; then
    echo "$RESULT"
    echo "[FAIL] -g WIF recovery succeeded on a non-CUDA build" >&2
    exit 1
fi

if grep -q "falling back to CPU" <<< "$RESULT"; then
    echo "$RESULT"
    echo "[FAIL] -g WIF recovery silently fell back to CPU" >&2
    exit 1
fi

if ! grep -q "GPU support not compiled" <<< "$RESULT"; then
    echo "$RESULT"
    echo "[FAIL] expected a clear GPU-not-compiled error" >&2
    exit 1
fi

echo "[PASS] non-CUDA build rejects -g WIF recovery without CPU fallback"
