#!/bin/bash
# Contract test: partial WIF recovery with -S must create/read BSGS cache files.

set -euo pipefail

REPO_ROOT=$(pwd)
KEYHUNT="$REPO_ROOT/keyhunt"

if [ ! -x "$KEYHUNT" ]; then
    echo "[ERROR] keyhunt binary not found; run make first" >&2
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

EXPECTED_WIF="KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFf9EZnLdvkn"
PARTIAL_WIF="KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFf*EZnLdvkn"
PUBKEY="02e963ffdfe34e63b68aeb42a5826e08af087660e0dac1c3e79f7625ca4e6ae482"

run_recovery() {
    timeout 60 "$KEYHUNT" -m wif-recovery -p "$PARTIAL_WIF" -P "$PUBKEY" -n 0x100000 -S -q -s 0 2>&1 || true
}

assert_success() {
    local result=$1
    local recovered
    if ! grep -q "SUCCESS" <<< "$result"; then
        echo "$result"
        echo "[FAIL] WIF recovery with -S did not report success" >&2
        exit 1
    fi

    recovered=$(awk '/Recovered WIF:/ {print $NF}' <<< "$result" | tail -n 1)
    if [ "$recovered" != "$EXPECTED_WIF" ]; then
        echo "$result"
        echo "[FAIL] Expected $EXPECTED_WIF, got ${recovered:-<empty>}" >&2
        exit 1
    fi
}

pushd "$TMPDIR" >/dev/null
RESULT=$(run_recovery)
assert_success "$RESULT"

if ! compgen -G 'keyhunt_bsgs_*' >/dev/null; then
    echo "$RESULT"
    echo "[FAIL] -S did not create BSGS cache files" >&2
    exit 1
fi

RESULT=$(run_recovery)
assert_success "$RESULT"
if ! grep -q "Reading .* from file keyhunt_bsgs_" <<< "$RESULT"; then
    echo "$RESULT"
    echo "[FAIL] second -S run did not read existing BSGS cache files" >&2
    exit 1
fi
popd >/dev/null

echo "[PASS] -S generated and reused BSGS cache files for WIF recovery"
