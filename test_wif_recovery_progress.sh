#!/bin/bash
# Regression test: CPU WIF-BSGS recovery should report progress during
# baby-step table population and A-side search.

set -euo pipefail

if [ ! -x ./keyhunt ]; then
    echo "[ERROR] keyhunt binary not found; run make first" >&2
    exit 1
fi

EXPECTED_WIF="KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFf9EZnLdvkn"
PARTIAL_WIF="KwDiBf8*QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFf*EZnLd*kn"
PUBKEY="02e963ffdfe34e63b68aeb42a5826e08af087660e0dac1c3e79f7625ca4e6ae482"

RESULT=$(timeout 120 ./keyhunt -m wif-recovery -p "$PARTIAL_WIF" -P "$PUBKEY" -n 0x100000 -q -s 0 2>&1 || true)

if ! grep -q "BSGS B steps progress:" <<< "$RESULT"; then
    echo "$RESULT"
    echo "[FAIL] BSGS B-step progress was not reported" >&2
    exit 1
fi

if ! grep -q "BSGS A steps progress:" <<< "$RESULT"; then
    echo "$RESULT"
    echo "[FAIL] BSGS A-step progress was not reported" >&2
    exit 1
fi

RECOVERED=$(awk '/Recovered WIF:/ {print $NF}' <<< "$RESULT" | tail -n 1)
if [ "$RECOVERED" != "$EXPECTED_WIF" ]; then
    echo "$RESULT"
    echo "[FAIL] Expected $EXPECTED_WIF, got ${RECOVERED:-<empty>}" >&2
    exit 1
fi

echo "[PASS] Progress reported and WIF recovered: $RECOVERED"
