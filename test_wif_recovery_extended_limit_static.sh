#!/bin/bash
# Static contract: WIF BSGS supports more than 10 missing chars without
# exceeding the current packed-combination limits.

set -euo pipefail

if ! grep -q '#define WIF_BSGS_MAX_TABLE_CHARS 5' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS table-side packed limit must remain explicit" >&2
    exit 1
fi

if ! grep -q '#define WIF_BSGS_MAX_SEARCH_CHARS 10' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS search-side packed limit must remain explicit" >&2
    exit 1
fi

if ! grep -q '#define WIF_BSGS_MAX_MISSING_CHARS (WIF_BSGS_MAX_TABLE_CHARS + WIF_BSGS_MAX_SEARCH_CHARS)' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS extended missing limit must derive from table/search limits" >&2
    exit 1
fi

if ! grep -q 'Need at least %d table chars to keep search chars <= %d' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS must reject extended searches that cannot fit a viable memory split" >&2
    exit 1
fi

if grep -q 'supports 1..10 missing' wif_recovery_cpu.cpp wif_recovery_cuda.cu; then
    echo "[FAIL] WIF BSGS still contains the old 10-character missing limit" >&2
    exit 1
fi

echo "[PASS] WIF BSGS extended missing-character limit is explicit"
