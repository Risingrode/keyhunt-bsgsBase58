#!/bin/bash
# Static performance contract for CUDA WIF BSGS.
#
# The A/B kernels must not recompute checksum correction points per
# combination. Host code should precompose each missing-character digit as:
#   digit*58^position*G contribution folded into the public-key contribution.

set -euo pipefail

if ! grep -q 'h_p_combined_b' wif_recovery_cuda.cu || ! grep -q 'h_p_combined_a' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS should precompute combined digit contribution tables" >&2
    exit 1
fi

if awk '
    /__global__ void wif_bsgs_build_b_kernel/,/^}/ {
        if ($0 ~ /device_compute_c_g/) bad = 1
        if ($0 ~ /device_point_x_prefix/) bad = 1
    }
    /__global__ void wif_bsgs_search_a_kernel/,/^}/ {
        if ($0 ~ /device_compute_c_g/) bad = 1
        if ($0 ~ /device_point_x_prefix/) bad = 1
    }
    /__global__ void wif_bsgs_probe_a_kernel/,/^}/ {
        if ($0 ~ /device_compute_c_g/) bad = 1
    }
    END { exit bad ? 0 : 1 }
' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS hot kernels still recompute checksum correction points" >&2
    exit 1
fi

if ! grep -q 'CUDA BSGS precomputed combined digit tables' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS should report combined-table precomputation" >&2
    exit 1
fi

if grep -q 'struct GpuBsgsEntry' wif_recovery_cuda.cu ||
   grep -q 'sizeof(GpuBsgsEntry)' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS table should use split x-prefix/packed arrays, not 16-byte entries" >&2
    exit 1
fi

if ! grep -Fq 'uint64_t *d_table_x_prefix' wif_recovery_cuda.cu ||
   ! grep -Fq 'uint32_t *d_table_packed' wif_recovery_cuda.cu ||
   ! grep -q 'bsgs_packed_c_value' wif_recovery_cuda.cu ||
   ! grep -q '12 B/slot' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS should compress the B table to x-prefix plus packed digits" >&2
    exit 1
fi

if ! grep -q 'point_add_affine(sum_p, raw_to_point(p_combined_b' wif_recovery_cuda.cu ||
   ! grep -q 'point_add_affine(sum_p, raw_to_point(p_combined_a' wif_recovery_cuda.cu ||
   ! grep -q 'point_add_affine(sum_p, raw_to_point(carry_g' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS hot kernels should use affine addition for precomputed points" >&2
    exit 1
fi

if ! grep -q 'block_batch_invert_zs' wif_recovery_cuda.cu ||
   ! grep -q 'block_batch_x_prefix' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS should batch-normalize projective points per block" >&2
    exit 1
fi

if ! grep -q 'wif_bsgs_build_b_kernel<<<b_blocks, threads_per_block, b_shared_bytes>>>' wif_recovery_cuda.cu ||
   ! grep -q 'wif_bsgs_search_a_kernel<<<a_blocks, threads_per_block, a_shared_bytes>>>' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS kernels should launch with dynamic shared memory for batch normalization" >&2
    exit 1
fi

if ! grep -q 'h_autotune_a_launch_config' wif_recovery_cuda.cu ||
   ! grep -q 'cudaEventElapsedTime' wif_recovery_cuda.cu ||
   ! grep -q 'CUDA autotune A config' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS should autotune A-side launch configuration on the selected GPU" >&2
    exit 1
fi

if ! grep -q 'a_chunk_combos' wif_recovery_cuda.cu ||
   ! grep -q 'CUDA A-side progress chunk' wif_recovery_cuda.cu ||
   ! grep -q 'print_cuda_progress(a_processed, a_combs' wif_recovery_cuda.cu; then
    echo "[FAIL] CUDA WIF BSGS should chunk A-side search and report live progress" >&2
    exit 1
fi

echo "[PASS] CUDA WIF BSGS uses compressed B table and precomputed combined digit tables"
