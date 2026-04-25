#!/bin/bash
# Static contract: -g WIF recovery should route to the GPU BSGS
# implementation, not the legacy CUDA checksum enumerator.

set -euo pipefail

if ! grep -q '#include "wif_recovery_bsgs_cuda.h"' keyhunt.cpp; then
    echo "[FAIL] keyhunt.cpp does not include the GPU BSGS WIF recovery header" >&2
    exit 1
fi

if ! grep -q 'cuda_wif_recovery_bsgs(' keyhunt.cpp; then
    echo "[FAIL] keyhunt.cpp does not call cuda_wif_recovery_bsgs for -g WIF recovery" >&2
    exit 1
fi

if grep -q 'Combinations per CUDA batch' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF path still exposes legacy full-combination batch enumeration" >&2
    exit 1
fi

echo "[PASS] -g WIF recovery is wired to GPU BSGS"
