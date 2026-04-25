#pragma once

#include "uint256.cuh"

/*
 * secp256k1 有限域 GF(P) 运算
 * P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
 * 使用 secp256k1 特有的快速规约: P = 2^256 - 2^32 - 977
 * 因此 2^256 ≡ 2^32 + 977 = 0x1000003D1 (mod P)
 */

/* ---- 域常量表达式（可在 host/device 代码中展开） ---- */

#define FIELD_P uint256( \
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, \
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL \
)

#define FIELD_P_MINUS_2 uint256( \
    0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, \
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL \
)

#define ORDER_N uint256( \
    0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, \
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL \
)

/* 规约常量: 2^32 + 977 = 0x1000003D1 */
static const uint64_t SECP256K1_C = 0x1000003D1ULL;

/* ---- 基本域运算 ---- */

/*
 * field_is_zero: 检查 a ≡ 0 (mod P)
 */
__device__ __forceinline__ bool field_is_zero(const uint256& a) {
    return a.is_zero() || a.is_equal(FIELD_P);
}

/*
 * field_is_equal: 检查 a ≡ b (mod P)
 */
__device__ __forceinline__ bool field_is_equal(const uint256& a, const uint256& b) {
    return a.is_equal(b);
}

/*
 * field_add: (a + b) mod P
 */
__device__ __forceinline__ uint256 field_add(const uint256& a, const uint256& b) {
    uint256 result;
    uint64_t carry = result.add_with_carry(a, b, 0);
    /* 如果有进位，结果肯定 >= P，需要减去 P */
    if (carry) {
        result.sub(FIELD_P);
    } else {
        /* 无进位，检查是否 >= P */
        uint256 tmp = result;
        uint64_t borrow = tmp.sub(FIELD_P);
        if (borrow == 0) {
            result = tmp; /* result >= P，使用减去P后的值 */
        }
        /* 否则 result < P，保持不变 */
    }
    return result;
}

/*
 * field_sub: (a - b) mod P
 */
__device__ __forceinline__ uint256 field_sub(const uint256& a, const uint256& b) {
    uint256 result;
    uint64_t borrow = result.sub_with_borrow(a, b, 0);
    if (borrow) {
        /* 借位说明结果为负，加上 P */
        result.add(FIELD_P);
    }
    return result;
}

/*
 * field_neg: (-a) mod P = (P - a) mod P
 */
__device__ __forceinline__ uint256 field_neg(const uint256& a) {
    if (a.is_zero()) return uint256(0, 0, 0, 0);
    uint256 result;
    result.sub_with_borrow(FIELD_P, a, 0);
    return result;
}

/*
 * field_double: (2 * a) mod P
 */
__device__ __forceinline__ uint256 field_double(const uint256& a) {
    return field_add(a, a);
}

/* ---- 核心: 256×256 乘法 + secp256k1 快速规约 ---- */

/*
 * 内部辅助: a + b + carry, 返回 {result, new_carry}
 * 使用 __uint128_t 确保进位不丢失
 */
__device__ __forceinline__ void add_carry128(
    uint64_t a, uint64_t b, uint64_t carry_in,
    uint64_t* result, uint64_t* carry_out)
{
    __uint128_t sum = (__uint128_t)a + b + carry_in;
    *result = (uint64_t)sum;
    *carry_out = (uint64_t)(sum >> 64);
}

/*
 * secp256k1_mod_reduce: 512位 → 256位 规约
 * 输入: r512[8] (512位乘积，little-endian 64位字)
 * 输出: 规约后的 uint256 (保证 < P)
 *
 * 算法: 使用 __uint128_t 追踪进位，确保不丢失
 * 1. 512→320位: 高256位 × 0x1000003D1 加到低256位
 * 2. 320→256位: 再次用 0x1000003D1
 * 3. 最终规约: 如果结果 >= P，减去 P
 */
__device__ __forceinline__ uint256 secp256k1_mod_reduce(uint64_t r512[8]) {
    uint64_t r[5] = {r512[0], r512[1], r512[2], r512[3], 0};
    uint64_t lo, hi, c;

    /* 步骤1: 512→320位规约 */
    /* 将 r512[4..7] × C 加到 r[0..4] */

    /* r512[4] × C */
    lo = r512[4] * SECP256K1_C;
    hi = __umul64hi(r512[4], SECP256K1_C);
    add_carry128(r[0], lo, 0, &r[0], &c);
    add_carry128(r[1], hi, c, &r[1], &c);
    add_carry128(r[2], 0, c, &r[2], &c);
    add_carry128(r[3], 0, c, &r[3], &c);
    r[4] += c;

    /* r512[5] × C */
    lo = r512[5] * SECP256K1_C;
    hi = __umul64hi(r512[5], SECP256K1_C);
    add_carry128(r[1], lo, 0, &r[1], &c);
    add_carry128(r[2], hi, c, &r[2], &c);
    add_carry128(r[3], 0, c, &r[3], &c);
    r[4] += c;

    /* r512[6] × C */
    lo = r512[6] * SECP256K1_C;
    hi = __umul64hi(r512[6], SECP256K1_C);
    add_carry128(r[2], lo, 0, &r[2], &c);
    add_carry128(r[3], hi, c, &r[3], &c);
    r[4] += c;

    /* r512[7] × C */
    lo = r512[7] * SECP256K1_C;
    hi = __umul64hi(r512[7], SECP256K1_C);
    add_carry128(r[3], lo, 0, &r[3], &c);
    r[4] += hi + c;

    /* 步骤2: 320→256位规约 */
    /* r[4] × C 加到 r[0..3] */
    lo = r[4] * SECP256K1_C;
    hi = __umul64hi(r[4], SECP256K1_C);
    add_carry128(r[0], lo, 0, &r[0], &c);
    add_carry128(r[1], hi, c, &r[1], &c);
    add_carry128(r[2], 0, c, &r[2], &c);
    add_carry128(r[3], 0, c, &r[3], &c);

    /* 步骤3: 最终规约 — 如果结果 >= P，减去 P */
    /* 概率极低 (~1/2^256)，但必须处理以保证正确性 */
    uint256 result(r[0], r[1], r[2], r[3]);
    uint256 tmp = result;
    uint64_t borrow = tmp.sub(FIELD_P);
    if (borrow == 0) {
        result = tmp; /* result >= P, 使用 result - P */
    }
    return result;
}

/*
 * field_mul: (a * b) mod P
 * 使用 __uint128_t 做 256×256→512 乘法 + secp256k1 快速规约
 *
 * 用 __uint128_t 追踪所有进位，确保正确性
 */
__device__ __forceinline__ uint256 field_mul(const uint256& a, const uint256& b) {
    uint64_t r512[8] = {0};

    /* 16次部分积，每次用 __uint128_t 正确累加 */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t lo_prod = a.v[i] * b.v[j];
            uint64_t hi_prod = __umul64hi(a.v[i], b.v[j]);

            __uint128_t sum = (__uint128_t)r512[i + j] + lo_prod + carry;
            r512[i + j] = (uint64_t)sum;
            carry = hi_prod + (uint64_t)(sum >> 64);
        }
        r512[i + 4] = carry;
    }

    return secp256k1_mod_reduce(r512);
}

/*
 * field_sqr: (a * a) mod P
 * 优化的平方运算
 */
__device__ __forceinline__ uint256 field_sqr(const uint256& a) {
    return field_mul(a, a);
}

/*
 * field_inv: a^(-1) mod P
 * 使用费马小定理: a^(-1) = a^(P-2) mod P
 * P-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
 *
 * 从最高有效位 (bit 255) 到最低位 (bit 0) 扫描
 * 需要约 256 次平方 + 约 128 次乘法
 */
__device__ __forceinline__ uint256 field_inv(const uint256& a) {
    uint256 result = UINT256_ONE;
    uint256 base = a;

    /* P-2 的 4 个 64位字 (little-endian: exp[0]=LSB, exp[3]=MSB) */
    const uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,  /* P-2 的 v[0] (LSB) */
        0xFFFFFFFFFFFFFFFFULL,  /* P-2 的 v[1] */
        0xFFFFFFFFFFFFFFFFULL,  /* P-2 的 v[2] */
        0xFFFFFFFFFFFFFFFFULL   /* P-2 的 v[3] (MSB) */
    };

    /* 从 MSB (word 3, bit 63) 到 LSB (word 0, bit 0) 扫描 */
    for (int word = 3; word >= 0; word--) {
        uint64_t e = exp[word];
        for (int bit = 63; bit >= 0; bit--) {
            result = field_sqr(result);
            if ((e >> bit) & 1ULL) {
                result = field_mul(result, base);
            }
        }
    }

    return result;
}

/* ---- 字节转换 ---- */

/*
 * field_from_bytes: 从大端字节数组构造域元素
 */
__device__ __forceinline__ uint256 field_from_bytes(const uint8_t bytes[32]) {
    uint256 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = (3 - i) * 8;
        result.v[i] = ((uint64_t)bytes[idx] << 56) |
                      ((uint64_t)bytes[idx + 1] << 48) |
                      ((uint64_t)bytes[idx + 2] << 40) |
                      ((uint64_t)bytes[idx + 3] << 32) |
                      ((uint64_t)bytes[idx + 4] << 24) |
                      ((uint64_t)bytes[idx + 5] << 16) |
                      ((uint64_t)bytes[idx + 6] << 8) |
                      ((uint64_t)bytes[idx + 7]);
    }
    return result;
}

/*
 * field_to_bytes: 域元素转大端字节数组
 */
__device__ __forceinline__ void field_to_bytes(const uint256& a, uint8_t bytes[32]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = (3 - i) * 8;
        bytes[idx]     = (uint8_t)(a.v[i] >> 56);
        bytes[idx + 1] = (uint8_t)(a.v[i] >> 48);
        bytes[idx + 2] = (uint8_t)(a.v[i] >> 40);
        bytes[idx + 3] = (uint8_t)(a.v[i] >> 32);
        bytes[idx + 4] = (uint8_t)(a.v[i] >> 24);
        bytes[idx + 5] = (uint8_t)(a.v[i] >> 16);
        bytes[idx + 6] = (uint8_t)(a.v[i] >> 8);
        bytes[idx + 7] = (uint8_t)(a.v[i]);
    }
}
