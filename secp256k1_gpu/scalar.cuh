#pragma once

#include "uint256.cuh"
#include "field.cuh"
#include "point.cuh"

/*
 * scalar.cuh — secp256k1 标量乘法
 *
 * 实现多种标量乘法策略:
 *   1. scalar_mul            — 标准 double-and-add
 *   2. scalar_mul_wnaf       — 窗口非相邻形式 (wNAF, w=4)
 *   3. scalar_mul_generator  — 生成元专用 (GTable 预计算表)
 *   4. precompute_table      — 批量预计算 table[i] = (i+1)*P
 *   5. scalar_mul_precomputed — 使用预计算表的标量乘法
 */

/* ================================================================
 * 1. 标准 double-and-add 标量乘法
 * ================================================================

 * 从最高有效位到最低位扫描:
 *   R = ∞
 *   for i = 255 downto 0:
 *       R = 2R
 *       if bit i of k is 1: R = R + P
 *   return R
 *
 * 最坏情况: 256 次倍乘 + 256 次加法 = 512 次群运算
 * 平均情况: 256 次倍乘 + 128 次加法 = 384 次群运算
 */
__device__ ECPoint scalar_mul(const ECPoint& P, const uint256& k) {
    ECPoint R;
    R.set_infinity();

    /* k == 0 或 P == ∞ 直接返回 ∞ */
    if (k.is_zero() || P.is_infinity()) return R;

    /* 从最高位 (bit 255) 到最低位 (bit 0) 扫描 */
    for (int i = 255; i >= 0; i--) {
        R = point_double(R);
        if (k.get_bit(i)) {
            R = point_add(R, P);
        }
    }
    return R;
}

/* ================================================================
 * 2. wNAF 窗口法标量乘法 (window size = 4)
 * ================================================================
 *
 * 预计算奇数倍: table[i] = (2i+1)*P, i = 0..7
 *   即: 1P, 3P, 5P, 7P, 9P, 11P, 13P, 15P
 *
 * wNAF 编码: 每个数字 d 满足:
 *   - d 是奇数或 0
 *   - |d| < 2^(w-1) = 8
 *   - 在任何 w 位窗口中最多只有一个非零数字
 *
 * 性能: 约 256 次倍乘 + 256/(w+1) ≈ 51 次加法
 * 加上 2^(w-1)-1 = 7 次预计算加法
 */

/*
 * 内部: 对标量进行 wNAF 编码
 *
 * wNAF (windowed Non-Adjacent Form) 保证:
 *   - 每个数字 d 是奇数或 0
 *   - |d| < 2^(w-1)
 *   - 任意连续 w 位中最多只有一个非零数字
 *
 * 算法 (从低位到高位):
 *   1. 如果最低位为 0: digit=0, scalar >>= 1
 *   2. 如果最低位为 1: 提取 w 位窗口
 *      - window < 2^(w-1): digit=window, scalar >>= w
 *      - window >= 2^(w-1): digit=window-2^w (负数),
 *        scalar = (scalar >> w) + 1 (进位)
 *
 * 注意: 进位加在 bit w 位置，不影响当前 w 位窗口的提取
 */
__device__ void wnaf_encode(int digits[257], const uint256& k, int w) {
    uint256 scalar = k;
    int i = 0;
    int limit = 1 << (w - 1);   /* 2^(w-1) */

    while (!scalar.is_zero()) {
        if (scalar.get_bit(0)) {
            /* 最低位为 1: 提取 w 位窗口 */
            int window = 0;
            for (int j = 0; j < w && (i + j) < 256; j++) {
                window |= (scalar.get_bit(j) << j);
            }

            if (window >= limit) {
                digits[i] = window - (1 << w);  /* 负数 digit */
                /* 进位: 在 bit w 位置加 1 */
                uint256 carry = uint256_ONE;
                for (int s = 0; s < w; s++) {
                    carry.shift_left_1();
                }
                scalar.add(carry);
            } else {
                digits[i] = window;  /* 正数 digit */
            }
            /* 消耗了 w 位 */
            for (int s = 0; s < w; s++) scalar.shift_right_1();
            i += w;
        } else {
            /* 最低位为 0: digit=0, 只消耗 1 位 */
            digits[i] = 0;
            scalar.shift_right_1();
            i++;
        }
    }

    /* 填充剩余位为 0 */
    while (i < 257) {
        digits[i++] = 0;
    }
}

/*
 * wNAF 标量乘法: R = k * P
 */
__device__ ECPoint scalar_mul_wnaf(const ECPoint& P, const uint256& k) {
    ECPoint R;
    R.set_infinity();

    if (k.is_zero() || P.is_infinity()) return R;

    /* 预计算奇数倍: table[i] = (2i+1)*P */
    ECPoint table[8];
    table[0] = P;                                 /* 1P */
    ECPoint P2 = point_double(P);                 /* 2P */
    for (int i = 1; i < 8; i++) {
        table[i] = point_add(table[i - 1], P2);  /* (2i+1)*P */
    }

    /* wNAF 编码 */
    int digits[257];
    wnaf_encode(digits, k, 4);

    /*
     * 从最高位到最低位扫描 wNAF digits
     * 优化: 跳过前导零 (leading zeros)，避免无意义的倍乘
     *
     * 流程:
     *   - 遇到前导零: 跳过 (不做倍乘)
     *   - 遇到第一个非零位: 直接加/减 (不做倍乘)
     *   - 之后每一位: 先倍乘，再根据 digit 加/减
     */
    ECPoint result;
    result.set_infinity();
    bool started = false;

    for (int i = 256; i >= 0; i--) {
        if (digits[i] != 0) {
            if (!started) {
                /* 第一个非零位: 直接赋值 (result 为 ∞, 不需要倍乘) */
                started = true;
            } else {
                /* 非第一个非零位: 先倍乘 */
                result = point_double(result);
            }

            if (digits[i] > 0) {
                result = point_add(result, table[digits[i] / 2]);
            } else {
                /* 负数: 加上其取反 */
                ECPoint neg_point = point_neg(table[(-digits[i]) / 2]);
                result = point_add(result, neg_point);
            }
        } else if (started) {
            /* digit == 0 但已经开始: 只做倍乘 */
            result = point_double(result);
        }
        /* digit == 0 且未开始: 跳过 (前导零) */
    }

    return result;
}

/* ================================================================
 * 3. 生成元乘法优化 (GTable 预计算表)
 * ================================================================
 *
 * GTable[i][j] = j * 256^i * G, 其中 i = 0..31, j = 0..255
 *
 * 标量 k 分解为 32 个字节: k = Σ k_i * 256^i
 * 则 k*G = Σ GTable[i][k_i]
 *
 * 最多 32 次点加法（跳过 k_i=0 的字节）
 *
 * GTable 在外部以 __constant__ 或 __device__ 内存分配
 */

/* 预计算表大小常量 */
#define GTABLE_LEVELS 32
#define GTABLE_ENTRIES 256

/*
 * 构建 GTable:
 *   table[0][0] = ∞ (占位)
 *   table[0][j] = j * G         (j = 1..255)
 *   table[i][0] = ∞ (占位)
 *   table[i][j] = j * 256^i * G
 *
 * 构建策略:
 *   Level 0: table[0][j] = table[0][j-1] + G (255 次加法)
 *   Level i>0: table[i][0] = ∞
 *              table[i][1] = table[i-1][255] + base = 256^i * G
 *              table[i][j] = table[i][j-1] + table[i][1]
 *
 * 在 host 端调用，结果拷贝到 GPU __constant__ 内存
 */
__device__ void build_gtable(ECPoint table[GTABLE_LEVELS][GTABLE_ENTRIES]) {
    ECPoint G = get_generator();

    /* Level 0: table[0][j] = j * G */
    table[0][0].set_infinity();
    table[0][1] = G;
    for (int j = 2; j < GTABLE_ENTRIES; j++) {
        table[0][j] = point_add(table[0][j - 1], G);
    }

    /*
     * Level 1..31: table[i][1] = 256^i * G
     *
     * 递推关系: table[i][1] = table[i-1][255] + table[i-1][1]
     * 因为 table[i-1][255] = 255 * 256^(i-1) * G
     *     table[i-1][1]   = 1  * 256^(i-1) * G
     *     两者之和         = 256 * 256^(i-1) * G = 256^i * G
     */
    ECPoint base = G;  /* base = 256^0 * G = G，用于计算下一级的 base */
    for (int i = 1; i < GTABLE_LEVELS; i++) {
        table[i][0].set_infinity();

        /* table[i][1] = table[i-1][255] + base = 256^i * G */
        table[i][1] = point_add(table[i - 1][GTABLE_ENTRIES - 1], base);
        base = table[i][1];  /* 更新 base 为 256^i * G，供下一级使用 */

        /* table[i][j] = table[i][j-1] + table[i][1] */
        for (int j = 2; j < GTABLE_ENTRIES; j++) {
            table[i][j] = point_add(table[i][j - 1], table[i][1]);
        }
    }
}

/*
 * 使用 GTable 进行生成元标量乘法: R = k * G
 *
 * 假设 GTable 已在 GPU __constant__ 或 __device__ 内存中
 */
__device__ ECPoint scalar_mul_generator(
    const uint256& k,
    const ECPoint gtable[GTABLE_LEVELS][GTABLE_ENTRIES])
{
    ECPoint R;
    R.set_infinity();

    if (k.is_zero()) return R;

    /* 逐字节查表累加 */
    for (int i = 0; i < 32; i++) {
        uint8_t byte_val = k.get_byte(i);
        if (byte_val == 0) continue;

        if (R.is_infinity()) {
            R = gtable[i][byte_val];
        } else {
            R = point_add(R, gtable[i][byte_val]);
        }
    }

    return R;
}

/* ================================================================
 * 4. 批量预计算表
 * ================================================================
 *
 * 预计算 table[i] = (i+1) * P, for i = 0..255
 * 用于 WIF 恢复中为每个缺失位置预计算 base58 的候选点
 *
 * 构建方法: 迭代加法
 *   table[0] = P
 *   table[i] = table[i-1] + P
 * 共 255 次点加法
 */
__device__ void precompute_table(const ECPoint& P, ECPoint table[256]) {
    table[0] = P;  /* 1*P */
    for (int i = 1; i < 256; i++) {
        table[i] = point_add(table[i - 1], P);  /* (i+1)*P */
    }
}

/*
 * 使用预计算表进行标量乘法: R = k * P
 * 将 k 分解为字节: k = Σ k_i * 256^i
 * R = Σ table[k_i - 1] * 256^i (跳过 k_i=0)
 *
 * 需要配合 256^i 的倍乘 (8 次点倍乘 per level)
 */
__device__ ECPoint scalar_mul_precomputed(
    const ECPoint& P,
    const uint256& k,
    const ECPoint table[256])
{
    ECPoint R;
    R.set_infinity();

    if (k.is_zero() || P.is_infinity()) return R;

    /* 从最高字节到最低字节处理 */
    for (int i = 31; i >= 0; i--) {
        /* 对 R 做 256 倍乘 (8 次倍乘) */
        for (int d = 0; d < 8; d++) {
            R = point_double(R);
        }

        uint8_t byte_val = k.get_byte(i);
        if (byte_val == 0) continue;

        /* R += byte_val * P */
        R = point_add(R, table[byte_val - 1]);
    }

    return R;
}

/*
 * 批量预计算仿射点表 (使用 Z=1 的 Jacobian 坐标)
 *
 * 与 precompute_table 相同，但结果自动为仿射表示 (z=1)
 * 适用于后续与仿射点进行混合加法的场景
 */
__device__ void precompute_table_affine(const ECPoint& P, ECPoint table[256]) {
    /* 先用 Jacobian 坐标迭代计算 */
    precompute_table(P, table);

    /* 将所有结果约化到仿射坐标 */
    for (int i = 0; i < 256; i++) {
        if (!table[i].is_infinity()) {
            table[i] = point_reduce(table[i]);
        }
    }
}
