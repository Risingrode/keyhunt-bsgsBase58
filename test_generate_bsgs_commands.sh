#!/bin/bash
# Verify the WIF/public-key test-case generator also emits runnable commands.

set -euo pipefail

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
OUTPUT_LOG="$TMPDIR/generator_output.txt"

REPO_ROOT=$PWD
(
    cd "$TMPDIR"
    python3 "$REPO_ROOT/generate_bsgs_test_case.py" --seed 1 --n 0x100000 --output-dir "$TMPDIR" >"$OUTPUT_LOG"
)

if [ ! -s "$TMPDIR/test_cases.txt" ]; then
    cat "$OUTPUT_LOG"
    echo "[FAIL] test_cases.txt was not generated" >&2
    exit 1
fi

if [ ! -x "$TMPDIR/test_commands.sh" ]; then
    ls -la "$TMPDIR"
    echo "[FAIL] executable test_commands.sh was not generated" >&2
    exit 1
fi

if ! grep -q "./keyhunt -m wif-recovery" "$TMPDIR/test_commands.sh"; then
    cat "$TMPDIR/test_commands.sh"
    echo "[FAIL] generated commands do not run WIF recovery" >&2
    exit 1
fi

if ! grep -q -- "-p '" "$TMPDIR/test_commands.sh" || ! grep -q -- "-P " "$TMPDIR/test_commands.sh"; then
    cat "$TMPDIR/test_commands.sh"
    echo "[FAIL] generated commands do not include partial WIF and pubkey" >&2
    exit 1
fi

if ! grep -q -- "-g" "$TMPDIR/test_commands.sh"; then
    cat "$TMPDIR/test_commands.sh"
    echo "[FAIL] generated commands do not include the GPU recovery command" >&2
    exit 1
fi

echo "[PASS] generator emitted test cases and runnable commands in $TMPDIR"
