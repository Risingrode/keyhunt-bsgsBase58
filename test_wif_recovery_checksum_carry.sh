#!/bin/bash
# Regression: this partial WIF requires a checksum low-32 carry of 1.
# The A-side BSGS target must subtract carry * 2^32*G to recover it.

set -euo pipefail

expected="KyhG3X3P2KdDqacKjSBeaQz1HFe8QwhzeUiphzqWdbvFLvgTwKbG"
output=$(
    ./keyhunt -m wif-recovery \
        -p 'KyhG3X3P2KdDqacKjSBe*Qz1HFe8QwhzeUiphzqWdbvFLvgTwKbG' \
        -P 020ec5b6b1cd49ab2273a1151cb8b88499e45e3b47d1a88a36c09032018e39e309 \
        -n 0x100000 -q -s 0 2>&1
)

if ! grep -q "Recovered WIF: ${expected}" <<<"${output}"; then
    echo "[FAIL] Carry=1 WIF recovery did not return the expected WIF" >&2
    echo "${output}" >&2
    exit 1
fi

echo "[PASS] Carry=1 WIF recovered: ${expected}"
