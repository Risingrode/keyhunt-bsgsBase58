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

if grep -R -q '\buint256_ONE\b' secp256k1_gpu wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA GPU math uses undefined uint256_ONE instead of UINT256_ONE" >&2
    exit 1
fi

if grep -R -q 'static const uint256 \\(UINT256_ZERO\\|UINT256_ONE\\|FIELD_P\\|FIELD_P_MINUS_2\\|ORDER_N\\)' secp256k1_gpu; then
    echo "[FAIL] CUDA GPU math constants must be device-visible expressions, not host static objects" >&2
    exit 1
fi

if grep -q 'static Point h_compute_c_g' wif_recovery_cuda.cu; then
    echo "[FAIL] Unused host h_compute_c_g helper should not be compiled into CUDA TU" >&2
    exit 1
fi

if grep -Eq 'Jacobian|z_inv2|z_inv3|Z1²|Z1³|Z2²|Z2³|X / Z²|Y / Z³' secp256k1_gpu/point.cuh; then
    echo "[FAIL] CUDA secp256k1 point math must stay in projective coordinates matching the CPU implementation" >&2
    exit 1
fi

if grep -q 'target = point_add(target, g_s)' wif_recovery_cuda.cu || grep -q 'Target = point_add(Target, G_S)' wif_recovery_cpu.cpp; then
    echo "[FAIL] BSGS checksum carry must subtract 2^32*G from the A-side target, not add it" >&2
    exit 1
fi

if awk '
    /goto cleanup_bsgs/ { seen_goto = 1 }
    /^cleanup_bsgs:/ { seen_goto = 0 }
    seen_goto && /(int threads_per_block =|int blocks =|time_t started_at =|int h_match_count =|int matches_to_copy =)/ { bad = 1 }
    END { exit bad ? 0 : 1 }
' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA BSGS cleanup gotos may bypass local variable initialization" >&2
    exit 1
fi

echo "[PASS] -g WIF recovery is wired to GPU BSGS"
