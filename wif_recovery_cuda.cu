/*
 * WIF Recovery using CUDA GPU
 * Recover missing characters in WIF private keys using CUDA checksum filtering.
 * keyhunt prepares the BSGS range/cache before this CUDA candidate filter runs.
 * 
 * Algorithm:
 * 1. Parse partial WIF, identify missing positions
 * 2. Generate all possible base58 combinations on GPU
 * 3. Verify checksum (fast filter, eliminates 99.999% combinations)
 * 4. Copy checksum-passing candidates back to the host for public-key verification
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <vector>

#include "wif_recovery_bsgs_cuda.h"
#include "secp256k1/SECP256k1.h"
#include "hash/sha256.h"
#include "secp256k1_gpu/point.cuh"

extern "C" int verify_privkey_pubkey(const uint8_t* privkey_bytes, const uint8_t* target_pubkey, int target_pubkey_len, int compressed);
extern Secp256K1 *secp;

#define WIF_BSGS_MAX_TABLE_CHARS 5
#define WIF_BSGS_MAX_SEARCH_CHARS 10
#define WIF_BSGS_MAX_MISSING_CHARS (WIF_BSGS_MAX_TABLE_CHARS + WIF_BSGS_MAX_SEARCH_CHARS)

// Base58 alphabet
__constant__ char d_base58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// SHA256 constants
__constant__ uint32_t d_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Result structure
struct WIFResult {
    uint64_t combination_index;
    char wif[64];
    int found;
    uint8_t privkey[32];
};

static int report_cuda_error(cudaError_t err, const char* operation) {
    if (err == cudaSuccess) {
        return 0;
    }
    fprintf(stderr, "[E] CUDA %s failed: %s\n", operation, cudaGetErrorString(err));
    return -1;
}

static void print_cuda_progress(
    uint64_t processed,
    uint64_t total,
    uint64_t launch_index,
    uint64_t total_launches,
    time_t started_at,
    const char *state
) {
    time_t now = time(NULL);
    double elapsed = difftime(now, started_at);
    double rate = elapsed > 0.0 ? (double)processed / elapsed : 0.0;
    double percent = total > 0 ? ((double)processed * 100.0) / (double)total : 100.0;
    double eta = rate > 0.0 && processed < total ? ((double)(total - processed) / rate) : 0.0;

    printf("[+] CUDA %s batch %" PRIu64 "/%" PRIu64
        ": %" PRIu64 "/%" PRIu64 " combinations (%.8f%%), %.0f combos/s, ETA %.0fs\n",
        state,
        launch_index + 1,
        total_launches,
        processed,
        total,
        percent,
        rate,
        eta);
    fflush(stdout);
}

static const char h_base58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

struct GpuUint256Raw {
    uint64_t v[4];
};

struct GpuPointRaw {
    GpuUint256Raw x;
    GpuUint256Raw y;
    GpuUint256Raw z;
};

struct GpuBsgsMatch {
    uint64_t packed_a;
    uint32_t packed_b;
    uint32_t c_a;
    uint32_t c_b;
    uint32_t carry;
};

struct GpuBsgsBProbe {
    uint64_t x_prefix;
    uint32_t c_value;
    uint32_t packed;
    int found;
};

struct GpuBsgsAProbe {
    uint64_t combo;
    uint64_t x_prefix[3];
    uint32_t c_value;
    uint64_t packed;
};

static int h_base58_value(char c) {
    for (int i = 0; i < 58; i++) {
        if (h_base58[i] == c) return i;
    }
    return 0;
}

static void h_double_sha256(const uint8_t *data, size_t len, uint8_t *hash) {
    uint8_t first[32];
    sha256((uint8_t*)data, len, first);
    sha256(first, 32, hash);
}

static int h_verify_wif_checksum(const char *wif, uint8_t *privkey_out) {
    uint8_t decoded[64];
    int decoded_len = 0;
    int leading_zeros = 0;
    int len = (int)strlen(wif);

    while (leading_zeros < len && wif[leading_zeros] == '1') {
        leading_zeros++;
    }

    for (int i = leading_zeros; i < len; i++) {
        int digit = h_base58_value(wif[i]);
        int carry = digit;
        for (int j = 0; j < decoded_len; j++) {
            int val = (int)decoded[j] * 58 + carry;
            decoded[j] = val & 0xff;
            carry = val >> 8;
        }
        while (carry > 0) {
            decoded[decoded_len++] = carry & 0xff;
            carry >>= 8;
        }
    }

    uint8_t full[64];
    int full_len = 0;
    for (int i = 0; i < leading_zeros; i++) full[full_len++] = 0;
    for (int i = 0; i < decoded_len; i++) full[full_len++] = decoded[decoded_len - 1 - i];

    if (full_len < 37) return 0;

    uint8_t hash[32];
    h_double_sha256(full, full_len - 4, hash);

    if (hash[0] == full[full_len - 4] && hash[1] == full[full_len - 3] &&
        hash[2] == full[full_len - 2] && hash[3] == full[full_len - 1]) {
        if (privkey_out) {
            for (int i = 0; i < 32; i++) privkey_out[i] = full[1 + i];
        }
        return 1;
    }
    return 0;
}

static bool h_point_is_infinity(Point &p) {
    return p.z.IsZero() || (p.x.IsZero() && p.y.IsZero());
}

static Point h_point_neg(Point &p) {
    Point zero;
    zero.Clear();
    if (h_point_is_infinity(p)) return zero;
    Point reduced = p;
    reduced.Reduce();
    return secp->Negation(reduced);
}

static Point h_point_add(Point &a, Point &b) {
    if (h_point_is_infinity(a)) return b;
    if (h_point_is_infinity(b)) return a;
    Point r = secp->Add(a, b);
    if (r.z.IsZero()) r.Clear();
    return r;
}

static Point h_point_double(Point &p) {
    Point zero;
    zero.Clear();
    if (h_point_is_infinity(p)) return zero;
    Point r = secp->Double(p);
    if (r.z.IsZero()) r.Clear();
    return r;
}

static Point h_multiply_g(Int &scalar) {
    Point zero;
    zero.Clear();
    if (scalar.IsZero()) return zero;
    return secp->ComputePublicKey(&scalar);
}

static uint64_t h_point_x_prefix(Point p) {
    if (h_point_is_infinity(p)) return 1ULL;
    p.Reduce();
    uint64_t x = p.x.bits64[0];
    return x == 0 ? 1ULL : x;
}

static Point h_lookup_c_g_from_table(uint32_t c, Point table[4][256]) {
    Point p;
    p.Clear();
    for (int i = 0; i < 4; i++) {
        uint8_t byte = (c >> (i * 8)) & 0xFF;
        if (byte != 0) {
            p = h_point_add(p, table[i][byte]);
        }
    }
    return p;
}

static GpuPointRaw h_point_to_raw(Point p) {
    GpuPointRaw out;
    if (h_point_is_infinity(p)) {
        memset(&out, 0, sizeof(out));
        return out;
    }
    p.Reduce();
    for (int i = 0; i < 4; i++) {
        out.x.v[i] = p.x.bits64[i];
        out.y.v[i] = p.y.bits64[i];
        out.z.v[i] = (i == 0) ? 1ULL : 0ULL;
    }
    return out;
}

static void h_decode_combo_point(
    uint64_t combo,
    const std::vector<Point> &combined_points,
    const int *positions,
    int num_positions,
    int wif_len,
    const uint32_t *pow58_mod32,
    Point *carry_g,
    Point *sum_p,
    uint32_t *sum_c,
    uint64_t *packed
) {
    Point p;
    p.Clear();
    uint64_t c_wide = 0;
    uint64_t bits = 0;
    uint64_t temp = combo;

    for (int i = 0; i < num_positions; i++) {
        int d = (int)(temp % 58ULL);
        temp /= 58ULL;
        if (d > 0) {
            Point addend = combined_points[((size_t)i * 58) + d];
            p = h_point_add(p, addend);
        }
        uint32_t c_part = (uint32_t)((uint32_t)d * pow58_mod32[wif_len - 1 - positions[i]]);
        c_wide += c_part;
        bits |= ((uint64_t)d << (6 * i));
    }

    uint32_t local_carry = (uint32_t)(c_wide >> 32);
    if (local_carry > 0) {
        p = h_point_add(p, carry_g[local_carry]);
    }

    *sum_p = p;
    *sum_c = (uint32_t)c_wide;
    *packed = bits;
}

static void h_make_b_probe(
    uint64_t combo,
    const std::vector<Point> &points,
    const int *positions,
    int num_positions,
    int wif_len,
    const uint32_t *pow58_mod32,
    Point *carry_g,
    GpuBsgsBProbe *probe
) {
    Point sum_p;
    uint32_t sum_c = 0;
    uint64_t packed = 0;
    h_decode_combo_point(combo, points, positions, num_positions, wif_len, pow58_mod32, carry_g, &sum_p, &sum_c, &packed);

    probe->x_prefix = h_point_x_prefix(sum_p);
    probe->c_value = sum_c;
    probe->packed = (uint32_t)packed;
    probe->found = 0;
}

static void h_make_a_probe(
    uint64_t combo,
    const std::vector<Point> &points,
    const int *positions,
    int num_positions,
    int wif_len,
    const uint32_t *pow58_mod32,
    Point *carry_g,
    Point base_point,
    Point g_s,
    GpuBsgsAProbe *probe
) {
    Point sum_p;
    uint32_t sum_c = 0;
    uint64_t packed = 0;
    h_decode_combo_point(combo, points, positions, num_positions, wif_len, pow58_mod32, carry_g, &sum_p, &sum_c, &packed);

    Point neg_pa = h_point_neg(sum_p);
    Point target = h_point_add(base_point, neg_pa);
    Point neg_g_s = h_point_neg(g_s);

    probe->combo = combo;
    probe->c_value = sum_c;
    probe->packed = packed;
    for (uint32_t carry = 0; carry <= 2; carry++) {
        if (carry > 0) {
            target = h_point_add(target, neg_g_s);
        }
        probe->x_prefix[carry] = h_point_x_prefix(target);
    }
}

static bool h_build_candidate_wif(
    const char *partial_wif,
    const int *a_pos,
    int num_a,
    uint64_t packed_a,
    const int *b_pos,
    int num_b,
    uint32_t packed_b,
    char *candidate,
    size_t candidate_size
) {
    size_t len = strlen(partial_wif);
    if (len + 1 > candidate_size) return false;
    strcpy(candidate, partial_wif);
    for (int i = 0; i < num_a; i++) {
        int d = (int)((packed_a >> (6 * i)) & 0x3FULL);
        if (d < 0 || d >= 58) return false;
        candidate[a_pos[i]] = h_base58[d];
    }
    for (int i = 0; i < num_b; i++) {
        int d = (int)((packed_b >> (6 * i)) & 0x3FU);
        if (d < 0 || d >= 58) return false;
        candidate[b_pos[i]] = h_base58[d];
    }
    return true;
}

__device__ __forceinline__ uint256 raw_to_uint256(const GpuUint256Raw &raw) {
    return uint256(raw.v[0], raw.v[1], raw.v[2], raw.v[3]);
}

__device__ __forceinline__ ECPoint raw_to_point(const GpuPointRaw &raw) {
    ECPoint p;
    p.x = raw_to_uint256(raw.x);
    p.y = raw_to_uint256(raw.y);
    p.z = raw_to_uint256(raw.z);
    return p;
}

__device__ __forceinline__ GpuPointRaw point_to_raw(const ECPoint &p) {
    GpuPointRaw raw;
    raw.x.v[0] = p.x.v[0]; raw.x.v[1] = p.x.v[1]; raw.x.v[2] = p.x.v[2]; raw.x.v[3] = p.x.v[3];
    raw.y.v[0] = p.y.v[0]; raw.y.v[1] = p.y.v[1]; raw.y.v[2] = p.y.v[2]; raw.y.v[3] = p.y.v[3];
    raw.z.v[0] = p.z.v[0]; raw.z.v[1] = p.z.v[1]; raw.z.v[2] = p.z.v[2]; raw.z.v[3] = p.z.v[3];
    return raw;
}

__device__ __forceinline__ uint64_t device_point_x_prefix(ECPoint p) {
    if (p.is_infinity()) return 1ULL;
    ECPoint reduced = point_reduce(p);
    uint64_t x = reduced.x.v[0];
    return x == 0 ? 1ULL : x;
}

__device__ __forceinline__ void block_batch_store_xz(
    uint256 *xs,
    uint256 *zs,
    int index,
    const ECPoint &p
) {
    if (p.is_infinity()) {
        xs[index] = UINT256_ZERO;
        zs[index] = UINT256_ZERO;
        return;
    }

    xs[index] = p.x;
    zs[index] = p.z;
}

__device__ void block_batch_invert_zs(uint256 *zs, uint256 *products, int count) {
    if (threadIdx.x == 0) {
        uint256 product = UINT256_ONE;

        for (int i = 0; i < count; i++) {
            products[i] = product;
            if (!zs[i].is_zero()) {
                product = field_mul(product, zs[i]);
            }
        }

        uint256 inverse = field_inv(product);
        for (int i = count - 1; i >= 0; i--) {
            uint256 z = zs[i];
            if (!z.is_zero()) {
                uint256 inv_z = field_mul(inverse, products[i]);
                inverse = field_mul(inverse, z);
                zs[i] = inv_z;
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ uint64_t block_batch_x_prefix(
    const uint256 *xs,
    const uint256 *inv_zs,
    int index
) {
    uint256 inv_z = inv_zs[index];
    if (inv_z.is_zero()) return 1ULL;

    uint256 x = field_mul(xs[index], inv_z);
    uint64_t prefix = x.v[0];
    return prefix == 0 ? 1ULL : prefix;
}

__device__ __forceinline__ uint32_t bsgs_packed_c_value(
    uint32_t packed,
    const int *b_pos,
    int num_b,
    int wif_len,
    const uint32_t *pow58_mod32
) {
    uint32_t c_value = 0;
    for (int i = 0; i < num_b; i++) {
        uint32_t d = (packed >> (6 * i)) & 0x3FU;
        c_value += d * pow58_mod32[wif_len - 1 - b_pos[i]];
    }
    return c_value;
}

__device__ __forceinline__ uint64_t bsgs_load_x_prefix(const uint64_t *table_x_prefix, uint32_t idx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(&table_x_prefix[idx]);
#else
    return table_x_prefix[idx];
#endif
}

__device__ __forceinline__ uint32_t bsgs_load_packed(const uint32_t *table_packed, uint32_t idx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(&table_packed[idx]);
#else
    return table_packed[idx];
#endif
}

__device__ void bsgs_insert_entry(
    uint64_t *table_x_prefix,
    uint32_t *table_packed,
    uint32_t table_mask,
    uint64_t x_prefix,
    uint32_t packed
) {
    if (x_prefix == 0) x_prefix = 1;
    uint32_t idx = (uint32_t)x_prefix & table_mask;
    while (true) {
        unsigned long long old = atomicCAS(
            (unsigned long long*)&table_x_prefix[idx],
            0ULL,
            (unsigned long long)x_prefix
        );
        if (old == 0ULL) {
            table_packed[idx] = packed;
            return;
        }
        idx = (idx + 1) & table_mask;
    }
}

__global__ void wif_bsgs_build_b_kernel(
    uint64_t total,
    const GpuPointRaw *p_combined_b,
    const int *b_pos,
    int num_b,
    int wif_len,
    const uint32_t *pow58_mod32,
    const GpuPointRaw *carry_g,
    uint64_t *table_x_prefix,
    uint32_t *table_packed,
    uint32_t table_mask
) {
    extern __shared__ uint64_t shared_words[];
    int batch_points = blockDim.x;
    uint256 *shared_x = (uint256*)shared_words;
    uint256 *shared_z = shared_x + batch_points;
    uint256 *shared_products = shared_z + batch_points;

    uint64_t grid_stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t block_start = (uint64_t)blockIdx.x * blockDim.x;

    for (uint64_t tile_base = block_start; tile_base < total; tile_base += grid_stride) {
        uint64_t combo = tile_base + threadIdx.x;
        bool active = combo < total;
        ECPoint sum_p;
        sum_p.set_infinity();
        uint64_t sum_c_wide = 0;
        uint32_t packed = 0;
        uint64_t temp = combo;

        if (active) {
            for (int i = 0; i < num_b; i++) {
                int d = (int)(temp % 58ULL);
                temp /= 58ULL;
                if (d > 0) {
                    sum_p = point_add_affine(sum_p, raw_to_point(p_combined_b[(i * 58) + d]));
                }
                uint32_t c_part = (uint32_t)((uint32_t)d * pow58_mod32[wif_len - 1 - b_pos[i]]);
                sum_c_wide += c_part;
                packed |= ((uint32_t)d << (6 * i));
            }

            uint32_t local_carry = (uint32_t)(sum_c_wide >> 32);
            if (local_carry > 0) {
                sum_p = point_add_affine(sum_p, raw_to_point(carry_g[local_carry]));
            }
        }

        block_batch_store_xz(shared_x, shared_z, threadIdx.x, sum_p);
        __syncthreads();
        block_batch_invert_zs(shared_z, shared_products, batch_points);

        if (active) {
            uint64_t x_prefix = block_batch_x_prefix(shared_x, shared_z, threadIdx.x);
            bsgs_insert_entry(table_x_prefix, table_packed, table_mask, x_prefix, packed);
        }
        __syncthreads();
    }
}

__global__ void wif_bsgs_search_a_kernel(
    uint64_t start_combo,
    uint64_t total,
    const GpuPointRaw *p_combined_a,
    const int *a_pos,
    int num_a,
    int wif_len,
    const uint32_t *pow58_mod32,
    const GpuPointRaw *carry_g,
    GpuPointRaw base_point_raw,
    GpuPointRaw g_s_raw,
    uint32_t c_k,
    const uint64_t *table_x_prefix,
    const uint32_t *table_packed,
    uint32_t table_mask,
    const int *b_pos,
    int num_b,
    GpuBsgsMatch *matches,
    int *match_count,
    int max_matches
) {
    extern __shared__ uint64_t shared_words[];
    int points_per_combo = 3;
    int batch_points = blockDim.x * points_per_combo;
    uint256 *shared_x = (uint256*)shared_words;
    uint256 *shared_z = shared_x + batch_points;
    uint256 *shared_products = shared_z + batch_points;

    uint64_t grid_stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t block_start = (uint64_t)blockIdx.x * blockDim.x;
    ECPoint base_point = raw_to_point(base_point_raw);
    ECPoint g_s = raw_to_point(g_s_raw);
    ECPoint neg_g_s = point_neg(g_s);

    for (uint64_t tile_base = block_start; tile_base < total; tile_base += grid_stride) {
        uint64_t local_combo = tile_base + threadIdx.x;
        uint64_t combo = start_combo + local_combo;
        bool active = local_combo < total;
        ECPoint sum_p;
        sum_p.set_infinity();
        uint64_t sum_c_wide = 0;
        uint64_t packed = 0;
        uint64_t temp = combo;

        if (active) {
            for (int i = 0; i < num_a; i++) {
                int d = (int)(temp % 58ULL);
                temp /= 58ULL;
                if (d > 0) {
                    sum_p = point_add_affine(sum_p, raw_to_point(p_combined_a[(i * 58) + d]));
                }
                uint32_t c_part = (uint32_t)((uint32_t)d * pow58_mod32[wif_len - 1 - a_pos[i]]);
                sum_c_wide += c_part;
                packed |= ((uint64_t)d << (6 * i));
            }

            uint32_t local_carry = (uint32_t)(sum_c_wide >> 32);
            if (local_carry > 0) {
                sum_p = point_add_affine(sum_p, raw_to_point(carry_g[local_carry]));
            }
        }

        uint32_t sum_c = (uint32_t)sum_c_wide;
        ECPoint target;
        target.set_infinity();
        if (active) {
            ECPoint neg_pa = point_neg(sum_p);
            target = point_add(base_point, neg_pa);
        }

        int point_base = threadIdx.x * points_per_combo;
        for (uint32_t carry = 0; carry <= 2; carry++) {
            if (active && carry > 0) {
                target = point_add_affine(target, neg_g_s);
            }
            block_batch_store_xz(shared_x, shared_z, point_base + (int)carry, target);
        }
        __syncthreads();
        block_batch_invert_zs(shared_z, shared_products, batch_points);

        if (active) {
            for (uint32_t carry = 0; carry <= 2; carry++) {
                uint64_t search_x = block_batch_x_prefix(shared_x, shared_z, point_base + (int)carry);
                uint32_t idx = (uint32_t)search_x & table_mask;
                uint64_t slot_x = bsgs_load_x_prefix(table_x_prefix, idx);

                while (slot_x != 0) {
                    if (slot_x == search_x) {
                        uint32_t packed_b = bsgs_load_packed(table_packed, idx);
                        uint32_t c_b = bsgs_packed_c_value(packed_b, b_pos, num_b, wif_len, pow58_mod32);
                        uint64_t total_c = (uint64_t)c_k + (uint64_t)sum_c + (uint64_t)c_b;
                        if ((total_c >> 32) == carry) {
                            int out_idx = atomicAdd(match_count, 1);
                            if (out_idx < max_matches) {
                                matches[out_idx].packed_a = packed;
                                matches[out_idx].packed_b = packed_b;
                                matches[out_idx].c_a = sum_c;
                                matches[out_idx].c_b = c_b;
                                matches[out_idx].carry = carry;
                            }
                        }
                    }
                    idx = (idx + 1) & table_mask;
                    slot_x = bsgs_load_x_prefix(table_x_prefix, idx);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void wif_bsgs_probe_b_table_kernel(
    GpuBsgsBProbe *probes,
    int num_probes,
    const uint64_t *table_x_prefix,
    const uint32_t *table_packed,
    uint32_t table_mask,
    const int *b_pos,
    int num_b,
    int wif_len,
    const uint32_t *pow58_mod32
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_probes) return;

    uint64_t x_prefix = probes[i].x_prefix;
    uint32_t idx = (uint32_t)x_prefix & table_mask;
    int found = 0;
    uint64_t slot_x = bsgs_load_x_prefix(table_x_prefix, idx);

    while (slot_x != 0) {
        uint32_t packed = bsgs_load_packed(table_packed, idx);
        if (slot_x == x_prefix && packed == probes[i].packed) {
            uint32_t c_value = bsgs_packed_c_value(packed, b_pos, num_b, wif_len, pow58_mod32);
            if (c_value == probes[i].c_value) {
                found = 1;
                break;
            }
        }
        idx = (idx + 1) & table_mask;
        slot_x = bsgs_load_x_prefix(table_x_prefix, idx);
    }

    probes[i].found = found;
}

__global__ void wif_bsgs_probe_a_kernel(
    GpuBsgsAProbe *probes,
    int num_probes,
    const GpuPointRaw *p_combined_a,
    const int *a_pos,
    int num_a,
    int wif_len,
    const uint32_t *pow58_mod32,
    const GpuPointRaw *carry_g,
    GpuPointRaw base_point_raw,
    GpuPointRaw g_s_raw
) {
    int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_idx >= num_probes) return;

    uint64_t combo = probes[probe_idx].combo;
    ECPoint base_point = raw_to_point(base_point_raw);
    ECPoint g_s = raw_to_point(g_s_raw);
    ECPoint neg_g_s = point_neg(g_s);
    ECPoint sum_p;
    sum_p.set_infinity();
    uint64_t sum_c_wide = 0;
    uint64_t packed = 0;
    uint64_t temp = combo;

    for (int i = 0; i < num_a; i++) {
        int d = (int)(temp % 58ULL);
        temp /= 58ULL;
        if (d > 0) {
            sum_p = point_add_affine(sum_p, raw_to_point(p_combined_a[(i * 58) + d]));
        }
        uint32_t c_part = (uint32_t)((uint32_t)d * pow58_mod32[wif_len - 1 - a_pos[i]]);
        sum_c_wide += c_part;
        packed |= ((uint64_t)d << (6 * i));
    }

    uint32_t local_carry = (uint32_t)(sum_c_wide >> 32);
    if (local_carry > 0) {
        sum_p = point_add_affine(sum_p, raw_to_point(carry_g[local_carry]));
    }
    uint32_t sum_c = (uint32_t)sum_c_wide;
    ECPoint neg_pa = point_neg(sum_p);
    ECPoint target = point_add(base_point, neg_pa);

    probes[probe_idx].c_value = sum_c;
    probes[probe_idx].packed = packed;
    for (uint32_t carry = 0; carry <= 2; carry++) {
        if (carry > 0) {
            target = point_add(target, neg_g_s);
        }
        probes[probe_idx].x_prefix[carry] = device_point_x_prefix(target);
    }
}

static uint64_t h_pow58_u64(int exp);
static uint64_t h_bsgs_table_slots(uint64_t combinations);
static size_t h_bsgs_table_bytes_for_split(int table_chars);

static uint64_t h_div_ceil_u64(uint64_t value, uint64_t divisor) {
    return divisor == 0 ? 0 : ((value + divisor - 1) / divisor);
}

static size_t h_cuda_memory_reserve(size_t free_mem) {
    const size_t min_reserve = 512ULL * 1024ULL * 1024ULL;
    size_t proportional_reserve = free_mem / 8ULL;
    return proportional_reserve > min_reserve ? proportional_reserve : min_reserve;
}

static size_t h_cuda_bsgs_aux_bytes(int table_chars, int search_chars, int max_matches) {
    size_t table_points = table_chars > 0 ? (size_t)table_chars * 58ULL : 1ULL;
    size_t search_points = search_chars > 0 ? (size_t)search_chars * 58ULL : 1ULL;
    size_t probe_count = 5;

    return
        table_points * sizeof(GpuPointRaw) +
        search_points * sizeof(GpuPointRaw) +
        (WIF_BSGS_MAX_SEARCH_CHARS + 1ULL) * sizeof(GpuPointRaw) +
        (table_chars > 0 ? (size_t)table_chars : 1ULL) * sizeof(int) +
        (search_chars > 0 ? (size_t)search_chars : 1ULL) * sizeof(int) +
        64ULL * sizeof(uint32_t) +
        (size_t)max_matches * sizeof(GpuBsgsMatch) +
        sizeof(int) +
        probe_count * sizeof(GpuBsgsBProbe) +
        probe_count * sizeof(GpuBsgsAProbe);
}

static size_t h_cuda_batch_shared_bytes(int threads_per_block, int points_per_combo) {
    return (size_t)threads_per_block * (size_t)points_per_combo * 3ULL * sizeof(uint256);
}

static size_t h_cuda_max_shared_mem_per_block(const cudaDeviceProp &prop) {
    size_t limit = prop.sharedMemPerBlock;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    if ((size_t)prop.sharedMemPerBlockOptin > limit) {
        limit = prop.sharedMemPerBlockOptin;
    }
#endif
    return limit;
}

static int h_prepare_a_kernel_shared_memory(const cudaDeviceProp &prop, size_t shared_bytes) {
    if (shared_bytes <= (size_t)prop.sharedMemPerBlock) return 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    if (shared_bytes <= (size_t)prop.sharedMemPerBlockOptin) {
        cudaError_t err = cudaFuncSetAttribute(
            wif_bsgs_search_a_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_bytes
        );
        return report_cuda_error(err, "set CUDA BSGS A dynamic shared memory limit");
    }
#endif
    fprintf(stderr,
        "[E] CUDA BSGS A needs %.2f KiB shared memory but device exposes %.2f KiB per block\n",
        (double)shared_bytes / 1024.0,
        (double)h_cuda_max_shared_mem_per_block(prop) / 1024.0);
    return -1;
}

static uint64_t h_bsgs_split_score(int table_chars, int search_chars) {
    uint64_t table_combs = h_pow58_u64(table_chars);
    uint64_t search_combs = h_pow58_u64(search_chars);
    return table_combs + (3ULL * search_combs);
}

static int h_choose_table_chars_for_memory(
    int max_table_chars,
    int num_missing,
    size_t free_mem,
    size_t *chosen_table_bytes,
    size_t *chosen_aux_bytes,
    size_t *chosen_reserve_bytes,
    int max_matches
) {
    size_t reserve = h_cuda_memory_reserve(free_mem);
    int min_table_chars = num_missing > WIF_BSGS_MAX_SEARCH_CHARS ? num_missing - WIF_BSGS_MAX_SEARCH_CHARS : 0;
    int best_table_chars = -1;
    uint64_t best_score = UINT64_MAX;
    size_t best_table_bytes = 0;
    size_t best_aux_bytes = 0;

    for (int table_chars = min_table_chars; table_chars <= max_table_chars; table_chars++) {
        int search_chars = num_missing - table_chars;
        size_t table_bytes = h_bsgs_table_bytes_for_split(table_chars);
        size_t aux_bytes = h_cuda_bsgs_aux_bytes(table_chars, search_chars, max_matches);
        if (table_bytes + aux_bytes + reserve > free_mem) {
            continue;
        }

        uint64_t score = h_bsgs_split_score(table_chars, search_chars);
        if (best_table_chars < 0 || score < best_score) {
            best_table_chars = table_chars;
            best_score = score;
            best_table_bytes = table_bytes;
            best_aux_bytes = aux_bytes;
        }
    }

    if (best_table_chars >= 0) {
        if (chosen_table_bytes) *chosen_table_bytes = best_table_bytes;
        if (chosen_aux_bytes) *chosen_aux_bytes = best_aux_bytes;
        if (chosen_reserve_bytes) *chosen_reserve_bytes = reserve;
        return best_table_chars;
    }

    if (chosen_table_bytes) *chosen_table_bytes = 0;
    if (chosen_aux_bytes) *chosen_aux_bytes = 0;
    if (chosen_reserve_bytes) *chosen_reserve_bytes = reserve;
    return -1;
}

static int h_choose_threads_per_block(const cudaDeviceProp &prop) {
    int candidates[] = {128, 64, 32};
    for (int i = 0; i < 3; i++) {
        int threads = candidates[i];
        if (prop.maxThreadsPerBlock >= threads &&
            h_cuda_max_shared_mem_per_block(prop) >= h_cuda_batch_shared_bytes(threads, 3)) {
            return threads;
        }
    }
    if (prop.maxThreadsPerBlock >= 32) return 32;
    return prop.maxThreadsPerBlock > 0 ? prop.maxThreadsPerBlock : 32;
}

static int h_choose_cuda_blocks(const cudaDeviceProp &prop, uint64_t total, int threads_per_block) {
    int sm_count = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    uint64_t target_blocks = (uint64_t)sm_count * 16ULL;
    uint64_t min_blocks = (uint64_t)sm_count * 4ULL;
    uint64_t max_blocks = (uint64_t)(prop.maxGridSize[0] > 0 ? prop.maxGridSize[0] : 65535);
    uint64_t needed_blocks = h_div_ceil_u64(total, (uint64_t)threads_per_block);

    if (target_blocks < min_blocks) target_blocks = min_blocks;
    if (target_blocks > needed_blocks && needed_blocks > 0) target_blocks = needed_blocks;
    if (target_blocks > max_blocks) target_blocks = max_blocks;
    if (target_blocks == 0) target_blocks = 1;
    return (int)target_blocks;
}

struct HCudaLaunchConfig {
    int threads;
    int blocks;
    size_t shared_bytes;
    double sample_combos_per_sec;
};

static uint64_t h_choose_a_chunk_combos(
    uint64_t total,
    const HCudaLaunchConfig &config
) {
    uint64_t grid_stride = (uint64_t)config.blocks * (uint64_t)config.threads;
    uint64_t min_chunk = grid_stride > 0 ? grid_stride * 64ULL : 1048576ULL;
    if (min_chunk < 1048576ULL) min_chunk = 1048576ULL;

    uint64_t chunk = min_chunk;
    if (config.sample_combos_per_sec > 0.0) {
        double target_seconds = 30.0;
        double desired = config.sample_combos_per_sec * target_seconds;
        if (desired > (double)UINT64_MAX) {
            chunk = UINT64_MAX;
        } else if (desired > 0.0) {
            chunk = (uint64_t)desired;
        }
    }

    if (chunk < min_chunk) chunk = min_chunk;
    if (chunk > total) chunk = total;
    return chunk == 0 ? total : chunk;
}

static HCudaLaunchConfig h_autotune_a_launch_config(
    const cudaDeviceProp &prop,
    uint64_t total,
    const GpuPointRaw *d_p_combined_a,
    const int *d_a_pos,
    int num_a,
    int wif_len,
    const uint32_t *d_pow58_mod32,
    const GpuPointRaw *d_carry_g,
    GpuPointRaw base_point_raw,
    GpuPointRaw g_s_raw,
    uint32_t c_k,
    const uint64_t *d_table_x_prefix,
    const uint32_t *d_table_packed,
    uint32_t table_mask,
    const int *d_b_pos,
    int num_b,
    GpuBsgsMatch *d_matches,
    int *d_match_count,
    int max_matches
) {
    HCudaLaunchConfig best;
    best.threads = h_choose_threads_per_block(prop);
    best.blocks = h_choose_cuda_blocks(prop, total, best.threads);
    best.shared_bytes = h_cuda_batch_shared_bytes(best.threads, 3);
    best.sample_combos_per_sec = 0.0;

    int candidates[] = {256, 128, 64, 32};
    cudaEvent_t started;
    cudaEvent_t stopped;
    if (report_cuda_error(cudaEventCreate(&started), "create CUDA autotune start event") != 0) {
        return best;
    }
    if (report_cuda_error(cudaEventCreate(&stopped), "create CUDA autotune stop event") != 0) {
        cudaEventDestroy(started);
        return best;
    }

    int zero = 0;
    for (int i = 0; i < 4; i++) {
        int threads = candidates[i];
        if (threads > prop.maxThreadsPerBlock) continue;

        size_t shared_bytes = h_cuda_batch_shared_bytes(threads, 3);
        if (shared_bytes > h_cuda_max_shared_mem_per_block(prop)) continue;
        if (h_prepare_a_kernel_shared_memory(prop, shared_bytes) != 0) continue;

        int blocks = h_choose_cuda_blocks(prop, total, threads);
        uint64_t grid_stride = (uint64_t)blocks * (uint64_t)threads;
        uint64_t bench_combos = grid_stride * 8ULL;
        if (bench_combos < 1048576ULL) bench_combos = 1048576ULL;
        if (bench_combos > total) bench_combos = total;
        if (bench_combos == 0) continue;

        if (report_cuda_error(cudaMemcpy(d_match_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "clear CUDA autotune match count") != 0) {
            continue;
        }
        if (report_cuda_error(cudaEventRecord(started), "record CUDA autotune start event") != 0) {
            continue;
        }
        wif_bsgs_search_a_kernel<<<blocks, threads, shared_bytes>>>(
            0,
            bench_combos,
            d_p_combined_a,
            d_a_pos,
            num_a,
            wif_len,
            d_pow58_mod32,
            d_carry_g,
            base_point_raw,
            g_s_raw,
            c_k,
            d_table_x_prefix,
            d_table_packed,
            table_mask,
            d_b_pos,
            num_b,
            d_matches,
            d_match_count,
            max_matches
        );
        if (report_cuda_error(cudaGetLastError(), "launch CUDA autotune A kernel") != 0) {
            continue;
        }
        if (report_cuda_error(cudaEventRecord(stopped), "record CUDA autotune stop event") != 0) {
            continue;
        }
        if (report_cuda_error(cudaEventSynchronize(stopped), "synchronize CUDA autotune stop event") != 0) {
            continue;
        }

        float elapsed_ms = 0.0f;
        if (report_cuda_error(cudaEventElapsedTime(&elapsed_ms, started, stopped), "measure CUDA autotune A kernel") != 0) {
            continue;
        }
        if (elapsed_ms <= 0.0f) continue;

        double rate = ((double)bench_combos * 1000.0) / (double)elapsed_ms;
        printf("[+] CUDA autotune A candidate: threads=%d, blocks=%d, shared=%.2f KiB, %.0f combos/s\n",
            threads,
            blocks,
            (double)shared_bytes / 1024.0,
            rate);
        if (rate > best.sample_combos_per_sec) {
            best.threads = threads;
            best.blocks = blocks;
            best.shared_bytes = shared_bytes;
            best.sample_combos_per_sec = rate;
        }
    }

    cudaEventDestroy(started);
    cudaEventDestroy(stopped);
    if (report_cuda_error(cudaMemcpy(d_match_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "clear CUDA autotune match count") != 0) {
        return best;
    }

    printf("[+] CUDA autotune A config: threads=%d, blocks=%d, shared=%.2f KiB, sample=%.0f combos/s\n",
        best.threads,
        best.blocks,
        (double)best.shared_bytes / 1024.0,
        best.sample_combos_per_sec);
    return best;
}

static int h_select_cuda_device(
    int desired_table_chars,
    int num_missing,
    int max_matches,
    cudaDeviceProp *selected_prop,
    size_t *selected_free_mem,
    size_t *selected_total_mem,
    int *selected_table_chars,
    size_t *selected_table_bytes,
    size_t *selected_aux_bytes,
    size_t *selected_reserve_bytes
) {
    int device_count = 0;
    if (report_cuda_error(cudaGetDeviceCount(&device_count), "device count") != 0) return -1;
    if (device_count == 0) {
        fprintf(stderr, "[E] No CUDA devices found\n");
        return -1;
    }

    int best_device = -1;
    int best_table_chars = -1;
    int best_sm_count = -1;
    size_t best_free_mem = 0;
    size_t best_total_mem = 0;
    size_t best_table_bytes = 0;
    size_t best_aux_bytes = 0;
    size_t best_reserve_bytes = 0;
    cudaDeviceProp best_prop;
    memset(&best_prop, 0, sizeof(best_prop));

    for (int device = 0; device < device_count; device++) {
        cudaDeviceProp prop;
        if (report_cuda_error(cudaGetDeviceProperties(&prop, device), "device properties") != 0) return -1;
        if (report_cuda_error(cudaSetDevice(device), "set device for memory query") != 0) return -1;

        size_t free_mem = 0;
        size_t total_mem = 0;
        if (report_cuda_error(cudaMemGetInfo(&free_mem, &total_mem), "memory info") != 0) return -1;

        size_t table_bytes = 0;
        size_t aux_bytes = 0;
        size_t reserve_bytes = 0;
        int table_chars = h_choose_table_chars_for_memory(
            desired_table_chars,
            num_missing,
            free_mem,
            &table_bytes,
            &aux_bytes,
            &reserve_bytes,
            max_matches
        );

        bool viable = table_chars >= 0;
        bool better = false;
        if (viable) {
            if (best_device < 0) {
                better = true;
            } else if (table_chars > best_table_chars) {
                better = true;
            } else if (table_chars == best_table_chars && prop.multiProcessorCount > best_sm_count) {
                better = true;
            } else if (table_chars == best_table_chars && prop.multiProcessorCount == best_sm_count && free_mem > best_free_mem) {
                better = true;
            }
        }

        if (better) {
            best_device = device;
            best_table_chars = table_chars;
            best_sm_count = prop.multiProcessorCount;
            best_free_mem = free_mem;
            best_total_mem = total_mem;
            best_table_bytes = table_bytes;
            best_aux_bytes = aux_bytes;
            best_reserve_bytes = reserve_bytes;
            best_prop = prop;
        }
    }

    if (best_device < 0) {
        int min_table_chars = num_missing > WIF_BSGS_MAX_SEARCH_CHARS ? num_missing - WIF_BSGS_MAX_SEARCH_CHARS : 0;
        fprintf(stderr,
            "[E] No CUDA device has enough free memory for %d missing chars. Need at least %d table chars to keep search chars <= %d.\n",
            num_missing,
            min_table_chars,
            WIF_BSGS_MAX_SEARCH_CHARS);
        return -1;
    }
    if (report_cuda_error(cudaSetDevice(best_device), "set selected device") != 0) return -1;

    if (selected_prop) *selected_prop = best_prop;
    if (selected_free_mem) *selected_free_mem = best_free_mem;
    if (selected_total_mem) *selected_total_mem = best_total_mem;
    if (selected_table_chars) *selected_table_chars = best_table_chars;
    if (selected_table_bytes) *selected_table_bytes = best_table_bytes;
    if (selected_aux_bytes) *selected_aux_bytes = best_aux_bytes;
    if (selected_reserve_bytes) *selected_reserve_bytes = best_reserve_bytes;
    return best_device;
}

static uint64_t h_pow58_u64(int exp) {
    uint64_t value = 1;
    for (int i = 0; i < exp; i++) value *= 58ULL;
    return value;
}

static uint64_t h_bsgs_table_slots(uint64_t combinations) {
    uint64_t min_table_size = combinations + (combinations / 2);
    uint64_t table_size = 2;
    while (table_size < min_table_size) table_size <<= 1;
    return table_size;
}

static size_t h_bsgs_table_x_bytes(uint64_t slots) {
    return (size_t)slots * sizeof(uint64_t);
}

static size_t h_bsgs_table_packed_bytes(uint64_t slots) {
    return (size_t)slots * sizeof(uint32_t);
}

static size_t h_bsgs_table_bytes_for_slots(uint64_t slots) {
    return h_bsgs_table_x_bytes(slots) + h_bsgs_table_packed_bytes(slots);
}

static size_t h_bsgs_table_bytes_for_split(int table_chars) {
    uint64_t combinations = h_pow58_u64(table_chars);
    uint64_t slots = h_bsgs_table_slots(combinations);
    return h_bsgs_table_bytes_for_slots(slots);
}

extern "C" int cuda_wif_recovery_bsgs(
    const char* partial_wif,
    const int* missing_positions,
    int num_missing,
    const uint8_t* target_pubkey,
    int target_pubkey_len,
    int compressed,
    char* result_wif
) {
    if (partial_wif == NULL || missing_positions == NULL || target_pubkey == NULL || result_wif == NULL) {
        fprintf(stderr, "[E] CUDA BSGS WIF recovery received a null input\n");
        return -2;
    }
    if (target_pubkey_len != 33 && target_pubkey_len != 65) {
        fprintf(stderr, "[E] Invalid target public key length for CUDA BSGS WIF recovery: %d\n", target_pubkey_len);
        return -2;
    }
    if ((compressed && target_pubkey_len != 33) || (!compressed && target_pubkey_len != 65)) {
        fprintf(stderr, "[E] Public key compression does not match CUDA BSGS WIF mode\n");
        return -2;
    }
    if (num_missing <= 0 || num_missing > WIF_BSGS_MAX_MISSING_CHARS) {
        fprintf(stderr,
            "[E] CUDA BSGS WIF recovery supports 1..%d missing characters in this build\n",
            WIF_BSGS_MAX_MISSING_CHARS);
        return -2;
    }
    int max_matches = 4096;

    int wif_len = (int)strlen(partial_wif);
    if (wif_len <= 0 || wif_len >= 64) {
        fprintf(stderr, "[E] Invalid WIF length for CUDA BSGS recovery: %d\n", wif_len);
        return -2;
    }
    for (int i = 0; i < num_missing; i++) {
        if (missing_positions[i] < 0 || missing_positions[i] >= wif_len) {
            fprintf(stderr, "[E] Invalid missing WIF position for CUDA BSGS recovery: %d\n", missing_positions[i]);
            return -2;
        }
    }

    result_wif[0] = '\0';

    int pos_copy[64];
    for (int i = 0; i < num_missing; i++) pos_copy[i] = missing_positions[i];
    for (int i = 0; i < num_missing - 1; i++) {
        for (int j = i + 1; j < num_missing; j++) {
            if (pos_copy[i] < pos_copy[j]) {
                int tmp = pos_copy[i];
                pos_copy[i] = pos_copy[j];
                pos_copy[j] = tmp;
            }
        }
    }

    int max_num_b = num_missing < WIF_BSGS_MAX_TABLE_CHARS ? num_missing : WIF_BSGS_MAX_TABLE_CHARS;

    cudaDeviceProp cuda_prop;
    memset(&cuda_prop, 0, sizeof(cuda_prop));
    size_t cuda_free_mem = 0;
    size_t cuda_total_mem = 0;
    size_t planned_table_bytes = 0;
    size_t planned_aux_bytes = 0;
    size_t planned_reserve_bytes = 0;
    int selected_table_chars = 0;
    int selected_device = h_select_cuda_device(
        max_num_b,
        num_missing,
        max_matches,
        &cuda_prop,
        &cuda_free_mem,
        &cuda_total_mem,
        &selected_table_chars,
        &planned_table_bytes,
        &planned_aux_bytes,
        &planned_reserve_bytes
    );
    if (selected_device < 0) return -1;

    int num_b = selected_table_chars;
    int num_a = num_missing - num_b;
    if (num_a > WIF_BSGS_MAX_SEARCH_CHARS || num_b > WIF_BSGS_MAX_TABLE_CHARS) {
        fprintf(stderr, "[E] CUDA BSGS split is too large: A=%d B=%d\n", num_a, num_b);
        return -2;
    }

    int h_b_pos[64] = {0};
    int h_a_pos[64] = {0};
    for (int i = 0; i < num_b; i++) h_b_pos[i] = pos_copy[i];
    for (int i = 0; i < num_a; i++) h_a_pos[i] = pos_copy[num_b + i];

    printf("[+] WIF recovery GPU BSGS mode enabled. Missing: %d chars\n", num_missing);
    printf("[+] CUDA selected device %d: %s (cc %d.%d, SMs=%d)\n",
        selected_device,
        cuda_prop.name,
        cuda_prop.major,
        cuda_prop.minor,
        cuda_prop.multiProcessorCount);
    printf("[+] CUDA memory: %.2f GiB free / %.2f GiB total, %.2f GiB reserved\n",
        (double)cuda_free_mem / 1073741824.0,
        (double)cuda_total_mem / 1073741824.0,
        (double)planned_reserve_bytes / 1073741824.0);
    printf("[+] BSGS split: table chars=%d, search chars=%d\n", num_b, num_a);
    printf("[+] CUDA memory plan: table %.2f GiB, auxiliary %.2f MiB\n",
        (double)planned_table_bytes / 1073741824.0,
        (double)planned_aux_bytes / 1048576.0);
    if (num_b != max_num_b) {
        printf("[+] CUDA split planner selected %d/%d instead of max-table %d/%d\n",
            num_b,
            num_a,
            max_num_b,
            num_missing - max_num_b);
    }

    Point p_target;
    char pubhex[132] = {0};
    for (int i = 0; i < target_pubkey_len; i++) sprintf(pubhex + i * 2, "%02x", target_pubkey[i]);
    bool dummy_comp;
    if (!secp->ParsePublicKeyHex(pubhex, p_target, dummy_comp)) {
        fprintf(stderr, "[E] Failed to parse target public key for CUDA BSGS recovery\n");
        return -2;
    }

    uint32_t h_pow58_mod32[64];
    h_pow58_mod32[0] = 1;
    for (int i = 1; i < 64; i++) h_pow58_mod32[i] = h_pow58_mod32[i - 1] * 58U;

    Int v_known_mod(0);
    uint32_t c_k = 0;
    for (int i = 0; i < wif_len; i++) {
        int digit = 0;
        if (partial_wif[i] != '*' && partial_wif[i] != '?' && partial_wif[i] != '.') {
            digit = h_base58_value(partial_wif[i]);
        }
        c_k = c_k * 58U + (uint32_t)digit;
        v_known_mod.Mult(58);
        v_known_mod.Add((uint64_t)digit);
        v_known_mod.Mod(&secp->order);
    }

    Int vk = v_known_mod;
    Int low_part((uint64_t)c_k);
    if (vk.IsLower(&low_part)) {
        vk.Add(&secp->order);
    }
    vk.Sub(&low_part);

    Int const_s(0x80);
    int s_shifts = compressed ? 296 : 288;
    for (int i = 0; i < s_shifts; i++) {
        const_s.Add(&const_s);
        const_s.Mod(&secp->order);
    }
    if (compressed) {
        Int pow32(1);
        for (int i = 0; i < 32; i++) {
            pow32.Add(&pow32);
            pow32.Mod(&secp->order);
        }
        const_s.Add(&pow32);
        const_s.Mod(&secp->order);
    }

    int p_shifts = compressed ? 40 : 32;
    Point p_scaled = p_target;
    for (int i = 0; i < p_shifts; i++) p_scaled = h_point_double(p_scaled);

    Point p1 = h_multiply_g(vk);
    p1 = secp->Negation(p1);
    Point p2 = h_multiply_g(const_s);

    Point base_point = h_point_add(p_scaled, p1);
    base_point = h_point_add(base_point, p2);

    Int s_val(1);
    for (int i = 0; i < 32; i++) {
        s_val.Add(&s_val);
        s_val.Mod(&secp->order);
    }
    Point g_s = h_multiply_g(s_val);

    Point h_t_g[4][256];
    for (int i = 0; i < 4; i++) {
        h_t_g[i][0].Clear();
        for (int j = 1; j < 256; j++) {
            Int scalar((uint64_t)j);
            for (int k = 0; k < (i * 8); k++) {
                scalar.Add(&scalar);
                scalar.Mod(&secp->order);
            }
            h_t_g[i][j] = h_multiply_g(scalar);
        }
    }

    Point h_carry_g[WIF_BSGS_MAX_SEARCH_CHARS + 1];
    GpuPointRaw h_carry_g_raw[WIF_BSGS_MAX_SEARCH_CHARS + 1];
    h_carry_g[0].Clear();
    h_carry_g_raw[0] = h_point_to_raw(h_carry_g[0]);
    for (int i = 1; i <= WIF_BSGS_MAX_SEARCH_CHARS; i++) {
        Int scalar((uint64_t)i);
        for (int k = 0; k < 32; k++) {
            scalar.Add(&scalar);
            scalar.Mod(&secp->order);
        }
        h_carry_g[i] = h_multiply_g(scalar);
        h_carry_g_raw[i] = h_point_to_raw(h_carry_g[i]);
    }

    size_t b_points_count = num_b > 0 ? (size_t)num_b * 58 : 1;
    size_t a_points_count = num_a > 0 ? (size_t)num_a * 58 : 1;
    std::vector<GpuPointRaw> h_p_combined_b(b_points_count);
    std::vector<GpuPointRaw> h_p_combined_a(a_points_count);
    std::vector<Point> h_p_combined_b_points(b_points_count);
    std::vector<Point> h_p_combined_a_points(a_points_count);

    for (int i = 0; i < num_b; i++) {
        Int exp(1);
        int p = wif_len - 1 - h_b_pos[i];
        for (int k = 0; k < p; k++) {
            exp.Mult(58);
            exp.Mod(&secp->order);
        }
        Point zero;
        zero.Clear();
        h_p_combined_b_points[(size_t)i * 58] = zero;
        h_p_combined_b[(size_t)i * 58] = h_point_to_raw(zero);
        for (int d = 1; d < 58; d++) {
            Int digit_exp(&exp);
            digit_exp.Mult((uint64_t)d);
            digit_exp.Mod(&secp->order);
            Point p_d = h_multiply_g(digit_exp);
            uint32_t c_part = (uint32_t)((uint32_t)d * h_pow58_mod32[p]);
            Point c_p = h_lookup_c_g_from_table(c_part, h_t_g);
            c_p = h_point_neg(c_p);
            Point combined = h_point_add(p_d, c_p);
            h_p_combined_b_points[((size_t)i * 58) + d] = combined;
            h_p_combined_b[((size_t)i * 58) + d] = h_point_to_raw(combined);
        }
    }

    for (int i = 0; i < num_a; i++) {
        Int exp(1);
        int p = wif_len - 1 - h_a_pos[i];
        for (int k = 0; k < p; k++) {
            exp.Mult(58);
            exp.Mod(&secp->order);
        }
        Point zero;
        zero.Clear();
        h_p_combined_a_points[(size_t)i * 58] = zero;
        h_p_combined_a[(size_t)i * 58] = h_point_to_raw(zero);
        for (int d = 1; d < 58; d++) {
            Int digit_exp(&exp);
            digit_exp.Mult((uint64_t)d);
            digit_exp.Mod(&secp->order);
            Point p_d = h_multiply_g(digit_exp);
            uint32_t c_part = (uint32_t)((uint32_t)d * h_pow58_mod32[p]);
            Point c_p = h_lookup_c_g_from_table(c_part, h_t_g);
            c_p = h_point_neg(c_p);
            Point combined = h_point_add(p_d, c_p);
            h_p_combined_a_points[((size_t)i * 58) + d] = combined;
            h_p_combined_a[((size_t)i * 58) + d] = h_point_to_raw(combined);
        }
    }

    printf("[+] CUDA BSGS precomputed combined digit tables: B=%zu points, A=%zu points\n",
        h_p_combined_b.size(),
        h_p_combined_a.size());

    uint64_t b_combs = h_pow58_u64(num_b);
    uint64_t a_combs = h_pow58_u64(num_a);

    uint64_t b_probe_combos[5] = {0, 1, 57, b_combs / 2, b_combs > 0 ? b_combs - 1 : 0};
    uint64_t a_probe_combos[5] = {0, 1, 57, a_combs / 2, a_combs > 0 ? a_combs - 1 : 0};
    std::vector<GpuBsgsBProbe> h_b_probes;
    std::vector<GpuBsgsAProbe> h_a_expected_probes;
    std::vector<GpuBsgsAProbe> h_a_device_probes;
    for (int i = 0; i < 5; i++) {
        bool duplicate = false;
        for (int j = 0; j < i; j++) {
            if (b_probe_combos[i] == b_probe_combos[j]) duplicate = true;
        }
        if (!duplicate && b_probe_combos[i] < b_combs) {
            GpuBsgsBProbe probe;
            h_make_b_probe(b_probe_combos[i], h_p_combined_b_points, h_b_pos, num_b, wif_len, h_pow58_mod32, h_carry_g, &probe);
            h_b_probes.push_back(probe);
        }
    }
    for (int i = 0; i < 5; i++) {
        bool duplicate = false;
        for (int j = 0; j < i; j++) {
            if (a_probe_combos[i] == a_probe_combos[j]) duplicate = true;
        }
        if (!duplicate && a_probe_combos[i] < a_combs) {
            GpuBsgsAProbe probe;
            h_make_a_probe(a_probe_combos[i], h_p_combined_a_points, h_a_pos, num_a, wif_len, h_pow58_mod32, h_carry_g, base_point, g_s, &probe);
            h_a_expected_probes.push_back(probe);
            h_a_device_probes.push_back(probe);
        }
    }

    uint64_t table_size64 = h_bsgs_table_slots(b_combs);
    if (table_size64 > UINT32_MAX) {
        fprintf(stderr, "[E] CUDA BSGS table is too large for 32-bit indexing: %" PRIu64 " slots\n", table_size64);
        return -3;
    }
    uint32_t table_size = (uint32_t)table_size64;
    uint32_t table_mask = table_size - 1;
    size_t table_x_bytes = h_bsgs_table_x_bytes(table_size64);
    size_t table_packed_bytes = h_bsgs_table_packed_bytes(table_size64);
    size_t table_bytes = table_x_bytes + table_packed_bytes;

    printf("[+] CUDA BSGS hash table: %u slots, %.2f GiB (x %.2f GiB + packed %.2f GiB, 12 B/slot)\n",
        table_size,
        (double)table_bytes / 1073741824.0,
        (double)table_x_bytes / 1073741824.0,
        (double)table_packed_bytes / 1073741824.0);
    printf("[+] CUDA BSGS combinations: table=%" PRIu64 ", search=%" PRIu64 "\n", b_combs, a_combs);

    GpuPointRaw *d_p_combined_b = NULL;
    GpuPointRaw *d_p_combined_a = NULL;
    GpuPointRaw *d_carry_g = NULL;
    int *d_b_pos = NULL;
    int *d_a_pos = NULL;
    uint32_t *d_pow58_mod32 = NULL;
    uint64_t *d_table_x_prefix = NULL;
    uint32_t *d_table_packed = NULL;
    GpuBsgsMatch *d_matches = NULL;
    int *d_match_count = NULL;
    GpuBsgsBProbe *d_b_probes = NULL;
    GpuBsgsAProbe *d_a_probes = NULL;
    GpuBsgsMatch *h_matches = NULL;
    int zero = 0;
    int rc = -1;
    int threads_per_block = h_choose_threads_per_block(cuda_prop);
    size_t b_shared_bytes = h_cuda_batch_shared_bytes(threads_per_block, 1);
    size_t a_shared_bytes = h_cuda_batch_shared_bytes(threads_per_block, 3);
    int b_blocks = h_choose_cuda_blocks(cuda_prop, b_combs, threads_per_block);
    int a_blocks = h_choose_cuda_blocks(cuda_prop, a_combs, threads_per_block);
    time_t started_at = time(NULL);
    int h_match_count = 0;
    int matches_to_copy = 0;
    bool saw_point_candidates = false;
    HCudaLaunchConfig a_config = {0, 0, 0, 0.0};
    uint64_t a_chunk_combos = 0;
    uint64_t a_total_launches = 0;
    uint64_t a_processed = 0;
    uint64_t a_launch_index = 0;
    time_t a_started_at = 0;

    printf("[+] CUDA launch config: B blocks=%d, A blocks=%d, threads=%d, shared B=%.2f KiB, shared A=%.2f KiB\n",
        b_blocks,
        a_blocks,
        threads_per_block,
        (double)b_shared_bytes / 1024.0,
        (double)a_shared_bytes / 1024.0);

    if (report_cuda_error(cudaMalloc((void**)&d_p_combined_b, h_p_combined_b.size() * sizeof(GpuPointRaw)), "allocate B combined points") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_p_combined_a, h_p_combined_a.size() * sizeof(GpuPointRaw)), "allocate A combined points") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_carry_g, sizeof(h_carry_g_raw)), "allocate carry adjustment points") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_b_pos, (num_b > 0 ? num_b : 1) * sizeof(int)), "allocate B positions") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_a_pos, (num_a > 0 ? num_a : 1) * sizeof(int)), "allocate A positions") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_pow58_mod32, 64 * sizeof(uint32_t)), "allocate pow58 table") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_table_x_prefix, table_x_bytes), "allocate CUDA BSGS x-prefix table") != 0) {
        rc = -3;
        goto cleanup_bsgs;
    }
    if (report_cuda_error(cudaMalloc((void**)&d_table_packed, table_packed_bytes), "allocate CUDA BSGS packed table") != 0) {
        rc = -3;
        goto cleanup_bsgs;
    }
    if (report_cuda_error(cudaMalloc((void**)&d_matches, max_matches * sizeof(GpuBsgsMatch)), "allocate CUDA BSGS matches") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMalloc((void**)&d_match_count, sizeof(int)), "allocate CUDA BSGS match count") != 0) goto cleanup_bsgs;

    if (report_cuda_error(cudaMemcpy(d_p_combined_b, h_p_combined_b.data(), h_p_combined_b.size() * sizeof(GpuPointRaw), cudaMemcpyHostToDevice), "copy B combined points") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMemcpy(d_p_combined_a, h_p_combined_a.data(), h_p_combined_a.size() * sizeof(GpuPointRaw), cudaMemcpyHostToDevice), "copy A combined points") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMemcpy(d_carry_g, h_carry_g_raw, sizeof(h_carry_g_raw), cudaMemcpyHostToDevice), "copy carry adjustment points") != 0) goto cleanup_bsgs;
    if (num_b > 0 && report_cuda_error(cudaMemcpy(d_b_pos, h_b_pos, num_b * sizeof(int), cudaMemcpyHostToDevice), "copy B positions") != 0) goto cleanup_bsgs;
    if (num_a > 0 && report_cuda_error(cudaMemcpy(d_a_pos, h_a_pos, num_a * sizeof(int), cudaMemcpyHostToDevice), "copy A positions") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMemcpy(d_pow58_mod32, h_pow58_mod32, 64 * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy pow58 table") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMemset(d_table_x_prefix, 0, table_x_bytes), "clear CUDA BSGS x-prefix table") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaMemcpy(d_match_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "clear CUDA BSGS match count") != 0) goto cleanup_bsgs;

    printf("[+] CUDA BSGS building B table on GPU...\n");
    print_cuda_progress(0, b_combs, 0, 1, started_at, "starting BSGS B");
    wif_bsgs_build_b_kernel<<<b_blocks, threads_per_block, b_shared_bytes>>>(
        b_combs,
        d_p_combined_b,
        d_b_pos,
        num_b,
        wif_len,
        d_pow58_mod32,
        d_carry_g,
        d_table_x_prefix,
        d_table_packed,
        table_mask
    );
    if (report_cuda_error(cudaGetLastError(), "launch CUDA BSGS B kernel") != 0) goto cleanup_bsgs;
    if (report_cuda_error(cudaDeviceSynchronize(), "synchronize CUDA BSGS B kernel") != 0) goto cleanup_bsgs;
    print_cuda_progress(b_combs, b_combs, 0, 1, started_at, "finished BSGS B");

    if (!h_b_probes.empty()) {
        size_t b_probe_bytes = h_b_probes.size() * sizeof(GpuBsgsBProbe);
        if (report_cuda_error(cudaMalloc((void**)&d_b_probes, b_probe_bytes), "allocate CUDA BSGS B probes") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaMemcpy(d_b_probes, h_b_probes.data(), b_probe_bytes, cudaMemcpyHostToDevice), "copy CUDA BSGS B probes") != 0) goto cleanup_bsgs;
        wif_bsgs_probe_b_table_kernel<<<1, 32>>>(
            d_b_probes,
            (int)h_b_probes.size(),
            d_table_x_prefix,
            d_table_packed,
            table_mask,
            d_b_pos,
            num_b,
            wif_len,
            d_pow58_mod32
        );
        if (report_cuda_error(cudaGetLastError(), "launch CUDA BSGS B self-test") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaDeviceSynchronize(), "synchronize CUDA BSGS B self-test") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaMemcpy(h_b_probes.data(), d_b_probes, b_probe_bytes, cudaMemcpyDeviceToHost), "copy CUDA BSGS B self-test") != 0) goto cleanup_bsgs;
        for (size_t i = 0; i < h_b_probes.size(); i++) {
            if (h_b_probes[i].found == 0) {
                fprintf(stderr,
                    "[E] CUDA BSGS B self-test failed: expected table entry x=%016" PRIx64 " c=%08x packed=%08x was not found\n",
                    h_b_probes[i].x_prefix,
                    h_b_probes[i].c_value,
                    h_b_probes[i].packed);
                rc = -4;
                goto cleanup_bsgs;
            }
        }
        printf("[+] CUDA BSGS B self-test passed (%zu probes)\n", h_b_probes.size());
    }

    if (!h_a_device_probes.empty()) {
        size_t a_probe_bytes = h_a_device_probes.size() * sizeof(GpuBsgsAProbe);
        if (report_cuda_error(cudaMalloc((void**)&d_a_probes, a_probe_bytes), "allocate CUDA BSGS A probes") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaMemcpy(d_a_probes, h_a_device_probes.data(), a_probe_bytes, cudaMemcpyHostToDevice), "copy CUDA BSGS A probes") != 0) goto cleanup_bsgs;
        wif_bsgs_probe_a_kernel<<<1, 32>>>(
            d_a_probes,
            (int)h_a_device_probes.size(),
            d_p_combined_a,
            d_a_pos,
            num_a,
            wif_len,
            d_pow58_mod32,
            d_carry_g,
            h_point_to_raw(base_point),
            h_point_to_raw(g_s)
        );
        if (report_cuda_error(cudaGetLastError(), "launch CUDA BSGS A self-test") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaDeviceSynchronize(), "synchronize CUDA BSGS A self-test") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaMemcpy(h_a_device_probes.data(), d_a_probes, a_probe_bytes, cudaMemcpyDeviceToHost), "copy CUDA BSGS A self-test") != 0) goto cleanup_bsgs;
        for (size_t i = 0; i < h_a_device_probes.size(); i++) {
            if (h_a_device_probes[i].c_value != h_a_expected_probes[i].c_value ||
                h_a_device_probes[i].packed != h_a_expected_probes[i].packed ||
                h_a_device_probes[i].x_prefix[0] != h_a_expected_probes[i].x_prefix[0] ||
                h_a_device_probes[i].x_prefix[1] != h_a_expected_probes[i].x_prefix[1] ||
                h_a_device_probes[i].x_prefix[2] != h_a_expected_probes[i].x_prefix[2]) {
                fprintf(stderr,
                    "[E] CUDA BSGS A self-test failed at combo=%" PRIu64
                    ": host x=[%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64 "] c=%08x packed=%" PRIx64
                    ", device x=[%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64 "] c=%08x packed=%" PRIx64 "\n",
                    h_a_expected_probes[i].combo,
                    h_a_expected_probes[i].x_prefix[0],
                    h_a_expected_probes[i].x_prefix[1],
                    h_a_expected_probes[i].x_prefix[2],
                    h_a_expected_probes[i].c_value,
                    h_a_expected_probes[i].packed,
                    h_a_device_probes[i].x_prefix[0],
                    h_a_device_probes[i].x_prefix[1],
                    h_a_device_probes[i].x_prefix[2],
                    h_a_device_probes[i].c_value,
                    h_a_device_probes[i].packed);
                rc = -4;
                goto cleanup_bsgs;
            }
        }
        printf("[+] CUDA BSGS A self-test passed (%zu probes)\n", h_a_device_probes.size());
    }

    printf("[+] CUDA BSGS searching A side on GPU...\n");
    a_config = h_autotune_a_launch_config(
        cuda_prop,
        a_combs,
        d_p_combined_a,
        d_a_pos,
        num_a,
        wif_len,
        d_pow58_mod32,
        d_carry_g,
        h_point_to_raw(base_point),
        h_point_to_raw(g_s),
        c_k,
        d_table_x_prefix,
        d_table_packed,
        table_mask,
        d_b_pos,
        num_b,
        d_matches,
        d_match_count,
        max_matches
    );
    threads_per_block = a_config.threads;
    a_blocks = a_config.blocks;
    a_shared_bytes = a_config.shared_bytes;
    if (h_prepare_a_kernel_shared_memory(cuda_prop, a_shared_bytes) != 0) goto cleanup_bsgs;

    a_chunk_combos = h_choose_a_chunk_combos(a_combs, a_config);
    a_total_launches = h_div_ceil_u64(a_combs, a_chunk_combos);
    a_processed = 0;
    a_launch_index = 0;
    a_started_at = time(NULL);

    printf("[+] CUDA A-side progress chunk: %" PRIu64 " combinations, %" PRIu64 " launches\n",
        a_chunk_combos,
        a_total_launches);
    print_cuda_progress(0, a_combs, 0, a_total_launches, a_started_at, "starting BSGS A");

    while (a_processed < a_combs) {
        uint64_t chunk_combos = a_combs - a_processed;
        if (chunk_combos > a_chunk_combos) chunk_combos = a_chunk_combos;

        wif_bsgs_search_a_kernel<<<a_blocks, threads_per_block, a_shared_bytes>>>(
            a_processed,
            chunk_combos,
            d_p_combined_a,
            d_a_pos,
            num_a,
            wif_len,
            d_pow58_mod32,
            d_carry_g,
            h_point_to_raw(base_point),
            h_point_to_raw(g_s),
            c_k,
            d_table_x_prefix,
            d_table_packed,
            table_mask,
            d_b_pos,
            num_b,
            d_matches,
            d_match_count,
            max_matches
        );
        if (report_cuda_error(cudaGetLastError(), "launch CUDA BSGS A kernel") != 0) goto cleanup_bsgs;
        if (report_cuda_error(cudaDeviceSynchronize(), "synchronize CUDA BSGS A kernel") != 0) goto cleanup_bsgs;

        a_processed += chunk_combos;
        print_cuda_progress(a_processed, a_combs, a_launch_index, a_total_launches, a_started_at, "progress BSGS A");

        if (report_cuda_error(cudaMemcpy(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost), "copy CUDA BSGS match count") != 0) goto cleanup_bsgs;
        if (h_match_count > 0) {
            saw_point_candidates = true;
            matches_to_copy = h_match_count > max_matches ? max_matches : h_match_count;
            if (h_match_count > max_matches) {
                fprintf(stderr,
                    "[W] CUDA BSGS found %d point candidates near A combo %" PRIu64 "; verifying first %d\n",
                    h_match_count,
                    a_processed - chunk_combos,
                    max_matches);
            }
            h_matches = (GpuBsgsMatch*)malloc((size_t)matches_to_copy * sizeof(GpuBsgsMatch));
            if (h_matches == NULL) {
                fprintf(stderr, "[E] Failed to allocate host CUDA BSGS matches\n");
                goto cleanup_bsgs;
            }
            if (report_cuda_error(cudaMemcpy(h_matches, d_matches, (size_t)matches_to_copy * sizeof(GpuBsgsMatch), cudaMemcpyDeviceToHost), "copy CUDA BSGS matches") != 0) goto cleanup_bsgs;

            for (int i = 0; i < matches_to_copy; i++) {
                char candidate[128];
                uint8_t privkey[32];
                if (!h_build_candidate_wif(partial_wif, h_a_pos, num_a, h_matches[i].packed_a, h_b_pos, num_b, h_matches[i].packed_b, candidate, sizeof(candidate))) {
                    continue;
                }
                if (!h_verify_wif_checksum(candidate, privkey)) {
                    continue;
                }
                if (verify_privkey_pubkey(privkey, target_pubkey, target_pubkey_len, compressed)) {
                    strcpy(result_wif, candidate);
                    printf("[+] CUDA BSGS WIF exact match found: %s\n", result_wif);
                    rc = 0;
                    goto cleanup_bsgs;
                }
            }

            free(h_matches);
            h_matches = NULL;
            if (report_cuda_error(cudaMemcpy(d_match_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "clear CUDA BSGS match count") != 0) goto cleanup_bsgs;
        }
        a_launch_index++;
    }

    if (!saw_point_candidates) {
        printf("[-] CUDA BSGS found no point candidates\n");
    } else {
        printf("[-] CUDA BSGS point candidates did not pass WIF/public-key verification\n");
    }
    rc = 1;

cleanup_bsgs:
    if (d_p_combined_b != NULL) cudaFree(d_p_combined_b);
    if (d_p_combined_a != NULL) cudaFree(d_p_combined_a);
    if (d_carry_g != NULL) cudaFree(d_carry_g);
    if (d_b_pos != NULL) cudaFree(d_b_pos);
    if (d_a_pos != NULL) cudaFree(d_a_pos);
    if (d_pow58_mod32 != NULL) cudaFree(d_pow58_mod32);
    if (d_table_x_prefix != NULL) cudaFree(d_table_x_prefix);
    if (d_table_packed != NULL) cudaFree(d_table_packed);
    if (d_matches != NULL) cudaFree(d_matches);
    if (d_match_count != NULL) cudaFree(d_match_count);
    if (d_b_probes != NULL) cudaFree(d_b_probes);
    if (d_a_probes != NULL) cudaFree(d_a_probes);
    free(h_matches);
    return rc;
}

// SHA256 helper functions
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// SHA256 transform
__device__ void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t T1, T2;
    
    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) | (uint32_t)block[i*4+3];
    }
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        T1 = h + Sigma1(e) + Ch(e, f, g) + d_sha256_k[i] + W[i];
        T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    
    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// SHA256 for variable length input
__device__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint8_t block[64];
    int offset = 0;
    
    // Process full blocks
    while (offset + 64 <= len) {
        sha256_transform(state, data + offset);
        offset += 64;
    }
    
    // Prepare final block with padding
    int remaining = len - offset;
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < remaining; i++) block[i] = data[offset + i];
    block[remaining] = 0x80;
    
    // Add length in bits as big-endian 64-bit
    uint64_t bit_len = (uint64_t)len * 8;
    if (remaining >= 56) {
        sha256_transform(state, block);
        for (int i = 0; i < 64; i++) block[i] = 0;
    }
    block[63] = (uint8_t)(bit_len);
    block[62] = (uint8_t)(bit_len >> 8);
    block[61] = (uint8_t)(bit_len >> 16);
    block[60] = (uint8_t)(bit_len >> 24);
    block[59] = (uint8_t)(bit_len >> 32);
    block[58] = (uint8_t)(bit_len >> 40);
    block[57] = (uint8_t)(bit_len >> 48);
    block[56] = (uint8_t)(bit_len >> 56);
    
    sha256_transform(state, block);
    
    // Output hash
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (state[i] >> 24) & 0xff;
        hash[i*4+1] = (state[i] >> 16) & 0xff;
        hash[i*4+2] = (state[i] >> 8) & 0xff;
        hash[i*4+3] = state[i] & 0xff;
    }
}

// Double SHA256
__device__ void double_sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    uint8_t first_hash[32];
    sha256(data, len, first_hash);
    sha256(first_hash, 32, hash);
}

// Base58 decode (simplified for WIF)
__device__ bool base58_decode_wif(const char* input, int len, uint8_t* output, int* out_len) {
    // WIF format: [version(1)][privkey(32)][compress_flag(1 optional)][checksum(4)]
    // Total: 37 or 38 bytes decoded
    
    // Count leading '1's (represent zero bytes)
    int leading_zeros = 0;
    while (leading_zeros < len && input[leading_zeros] == '1') {
        leading_zeros++;
    }
    
    // Decode base58
    uint8_t decoded[64];
    int decoded_len = 0;
    
    for (int i = leading_zeros; i < len; i++) {
        char c = input[i];
        int digit = -1;
        
        // Find digit in base58 alphabet
        for (int j = 0; j < 58; j++) {
            if (d_base58[j] == c) {
                digit = j;
                break;
            }
        }
        
        if (digit == -1) return false; // Invalid character
        
        // Multiply by 58 and add digit
        int carry = digit;
        for (int j = 0; j < decoded_len; j++) {
            int val = (int)decoded[j] * 58 + carry;
            decoded[j] = val & 0xff;
            carry = val >> 8;
        }
        while (carry > 0) {
            if (decoded_len >= 64) return false;
            decoded[decoded_len++] = carry & 0xff;
            carry >>= 8;
        }
    }
    
    // Add leading zeros
    int total_len = leading_zeros + decoded_len;
    if (total_len > 64) return false;
    for (int i = 0; i < leading_zeros; i++) {
        output[i] = 0;
    }
    for (int i = 0; i < decoded_len; i++) {
        output[leading_zeros + i] = decoded[decoded_len - 1 - i];
    }
    
    *out_len = total_len;
    return true;
}

// Checksum verification (fast filter)
__device__ bool verify_checksum(const uint8_t* data, int data_len) {
    // data = [version + privkey + optional_compress_flag]
    // checksum = double_sha256(data)[0:4]
    // WIF = [data][checksum]
    
    if (data_len < 37 || data_len > 38) return false;
    
    uint8_t hash[32];
    double_sha256(data, data_len - 4, hash);
    
    // Compare checksum (last 4 bytes)
    return (hash[0] == data[data_len - 4]) &&
           (hash[1] == data[data_len - 3]) &&
           (hash[2] == data[data_len - 2]) &&
           (hash[3] == data[data_len - 1]);
}

__device__ bool is_canonical_wif_payload(const uint8_t* data, int data_len, int compressed) {
    if (data[0] != 0x80) return false;
    if (compressed) {
        return data_len == 38 && data[33] == 0x01;
    }
    return data_len == 37;
}

// Kernel to process WIF combinations
__global__ void wif_recovery_kernel(
    const char* partial_wif,
    int wif_len,
    const int* missing_positions,
    int num_missing,
    int compressed,
    uint64_t start_index,
    uint64_t total_combinations,
    uint64_t combinations_per_thread,
    WIFResult* results,
    int* result_count
) {
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_combo = start_index + thread_id * combinations_per_thread;
    uint64_t end_combo = start_combo + combinations_per_thread;
    
    char candidate_wif[64];
    uint8_t decoded[64];
    
    for (uint64_t combo = start_combo; combo < end_combo && combo < total_combinations; combo++) {
        // Generate candidate WIF
        for (int i = 0; i < wif_len; i++) {
            candidate_wif[i] = partial_wif[i];
        }
        candidate_wif[wif_len] = '\0';
        
        // Fill in missing positions
        uint64_t temp = combo;
        for (int i = 0; i < num_missing; i++) {
            int pos = missing_positions[i];
            int digit = temp % 58;
            temp /= 58;
            candidate_wif[pos] = d_base58[digit];
        }
        
        // Decode WIF
        int decoded_len;
        if (!base58_decode_wif(candidate_wif, wif_len, decoded, &decoded_len)) {
            continue;
        }
        
        // Verify checksum (fast filter)
        if (!verify_checksum(decoded, decoded_len)) {
            continue;
        }

        if (!is_canonical_wif_payload(decoded, decoded_len, compressed)) {
            continue;
        }
        
        // If checksum match found, save result to buffer
        int idx = atomicAdd(result_count, 1);
        if (idx < 1024) {
            results[idx].combination_index = combo;
            for (int i = 0; i <= wif_len; i++) {
                results[idx].wif[i] = candidate_wif[i];
            }
            for(int i = 0; i < 32; i++) {
                results[idx].privkey[i] = decoded[1 + i]; // skip version byte
            }
            results[idx].found = 1;
        }
    }
}

// Host function to launch CUDA kernel
extern "C" int cuda_wif_recovery(
    const char* partial_wif,
    const int* missing_positions,
    int num_missing,
    const uint8_t* target_pubkey,
    int target_pubkey_len,
    int compressed,
    char* result_wif
) {
    if (partial_wif == NULL || missing_positions == NULL || target_pubkey == NULL || result_wif == NULL) {
        fprintf(stderr, "[E] CUDA WIF recovery received a null input\n");
        return -2;
    }
    if (target_pubkey_len != 33 && target_pubkey_len != 65) {
        fprintf(stderr, "[E] Invalid target public key length for CUDA WIF recovery: %d\n", target_pubkey_len);
        return -2;
    }
    if ((compressed && target_pubkey_len != 33) || (!compressed && target_pubkey_len != 65)) {
        fprintf(stderr, "[E] Public key compression does not match CUDA WIF mode\n");
        return -2;
    }
    if (num_missing <= 0) {
        fprintf(stderr, "[E] CUDA WIF recovery needs at least one missing WIF character\n");
        return -2;
    }
    int wif_len = (int)strlen(partial_wif);
    if (wif_len <= 0 || wif_len >= 64) {
        fprintf(stderr, "[E] Invalid WIF length for CUDA recovery: %d\n", wif_len);
        return -2;
    }
    for (int i = 0; i < num_missing; i++) {
        if (missing_positions[i] < 0 || missing_positions[i] >= wif_len) {
            fprintf(stderr, "[E] Invalid missing WIF position for CUDA recovery: %d\n", missing_positions[i]);
            return -2;
        }
    }

    // Check CUDA device
    int device_count = 0;
    if (report_cuda_error(cudaGetDeviceCount(&device_count), "device count") != 0) {
        return -1;
    }
    if (device_count == 0) {
        fprintf(stderr, "[E] No CUDA devices found\n");
        return -1;
    }
    
    if (report_cuda_error(cudaSetDevice(0), "set device") != 0) {
        return -1;
    }
    
    // Calculate total combinations
    uint64_t total_combinations = 1;
    for (int i = 0; i < num_missing; i++) {
        if (total_combinations > UINT64_MAX / 58ULL) {
            fprintf(stderr, "[E] Too many combinations for 64-bit CUDA indexing\n");
            return -2;
        }
        total_combinations *= 58ULL;
    }
    
    printf("[+] Total combinations: %" PRIu64 "\n", total_combinations);
    
    // Allocate device memory
    char* d_partial_wif = NULL;
    int* d_missing_positions = NULL;
    WIFResult* d_results = NULL;
    int* d_result_count = NULL;
    WIFResult* h_results = NULL;
    int max_results = 1024;
    int zero = 0;
    int threads_per_block = 256;
    int blocks = 1024;
    int match_found = 0;
    int rc = -1;
    const uint64_t combinations_per_thread = 262144ULL;
    uint64_t combinations_per_launch = 0;
    uint64_t launch_start = 0;
    uint64_t launch_index = 0;
    uint64_t total_launches = 0;
    time_t progress_started_at = time(NULL);
    result_wif[0] = '\0';

    if (report_cuda_error(cudaMalloc((void**)&d_partial_wif, wif_len + 1), "allocate partial WIF") != 0) goto cleanup;
    if (report_cuda_error(cudaMalloc((void**)&d_missing_positions, num_missing * sizeof(int)), "allocate missing positions") != 0) goto cleanup;
    if (report_cuda_error(cudaMalloc((void**)&d_results, max_results * sizeof(WIFResult)), "allocate result buffer") != 0) goto cleanup;
    if (report_cuda_error(cudaMalloc((void**)&d_result_count, sizeof(int)), "allocate result count") != 0) goto cleanup;
    
    // Copy data to device
    if (report_cuda_error(cudaMemcpy(d_partial_wif, partial_wif, wif_len + 1, cudaMemcpyHostToDevice), "copy partial WIF") != 0) goto cleanup;
    if (report_cuda_error(cudaMemcpy(d_missing_positions, missing_positions, num_missing * sizeof(int), cudaMemcpyHostToDevice), "copy missing positions") != 0) goto cleanup;
    
    // Configure kernel launch. Keep each launch bounded so large WIF gaps
    // such as 58^8 are processed in batches instead of being rejected.
    combinations_per_launch = (uint64_t)blocks * (uint64_t)threads_per_block * combinations_per_thread;
    total_launches = (total_combinations + combinations_per_launch - 1) / combinations_per_launch;

    printf("[+] Launching CUDA batches: %d blocks, %d threads\n", blocks, threads_per_block);
    printf("[+] Combinations per thread: %" PRIu64 "\n", combinations_per_thread);
    printf("[+] Legacy CUDA checksum combinations per batch: %" PRIu64 "\n", combinations_per_launch);
    printf("[+] Total CUDA batches: %" PRIu64 "\n", total_launches);
    fflush(stdout);

    h_results = (WIFResult*)malloc(max_results * sizeof(WIFResult));
    if (h_results == NULL) {
        fprintf(stderr, "[E] Failed to allocate host result buffer\n");
        goto cleanup;
    }

    for (launch_start = 0; launch_start < total_combinations; launch_start += combinations_per_launch, launch_index++) {
        int h_result_count = 0;
        int results_to_copy = 0;

        if (report_cuda_error(cudaMemcpy(d_result_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "clear result count") != 0) goto cleanup;

        print_cuda_progress(
            launch_start,
            total_combinations,
            launch_index,
            total_launches,
            progress_started_at,
            "starting"
        );

        wif_recovery_kernel<<<blocks, threads_per_block>>>(
            d_partial_wif,
            wif_len,
            d_missing_positions,
            num_missing,
            compressed,
            launch_start,
            total_combinations,
            combinations_per_thread,
            d_results,
            d_result_count
        );

        if (report_cuda_error(cudaGetLastError(), "launch WIF recovery kernel") != 0) goto cleanup;
        if (report_cuda_error(cudaDeviceSynchronize(), "synchronize WIF recovery kernel") != 0) goto cleanup;

        {
            uint64_t processed = launch_start + combinations_per_launch;
            if (processed > total_combinations) processed = total_combinations;
            print_cuda_progress(
                processed,
                total_combinations,
                launch_index,
                total_launches,
                progress_started_at,
                "finished"
            );
        }

        if (report_cuda_error(cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost), "copy result count") != 0) goto cleanup;

        results_to_copy = (h_result_count > max_results) ? max_results : h_result_count;
        if (h_result_count > max_results) {
            fprintf(stderr, "[W] CUDA batch %" PRIu64 " found %d checksum candidates; verifying the first %d\n",
                launch_index, h_result_count, max_results);
        }

        if (results_to_copy > 0) {
            if (report_cuda_error(cudaMemcpy(h_results, d_results, results_to_copy * sizeof(WIFResult), cudaMemcpyDeviceToHost), "copy WIF candidates") != 0) {
                goto cleanup;
            }
        }

        if (h_result_count > 0) {
            printf("[+] GPU batch %" PRIu64 " found %d valid WIF checksum candidates. Verifying public keys...\n",
                launch_index, h_result_count);
        }

        for(int i = 0; i < results_to_copy; i++) {
            if(verify_privkey_pubkey(h_results[i].privkey, target_pubkey, target_pubkey_len, compressed)) {
                strcpy(result_wif, h_results[i].wif);
                printf("[+] WIF exact match found: %s\n", result_wif);
                match_found = 1;
                break;
            }
        }

        if (match_found) {
            rc = 0;
            goto cleanup;
        }

    }
    
    printf("[-] No match found\n");
    rc = 1;

cleanup:
    if (d_partial_wif != NULL) cudaFree(d_partial_wif);
    if (d_missing_positions != NULL) cudaFree(d_missing_positions);
    if (d_results != NULL) cudaFree(d_results);
    if (d_result_count != NULL) cudaFree(d_result_count);
    free(h_results);
    return rc;
}
