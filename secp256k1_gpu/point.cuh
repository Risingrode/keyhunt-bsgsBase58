#pragma once

#include "uint256.cuh"
#include "field.cuh"

/*
 * point.cuh — secp256k1 椭圆曲线 Jacobian 坐标点运算
 *
 * Jacobian 坐标 (X, Y, Z) 对应仿射坐标 (x, y) = (X/Z^2, Y/Z^3)
 * 当 Z = 0 时表示无穷远点（单位元）
 * secp256k1 曲线参数: a = 0, b = 7
 */

/* ==================== 点结构体 ==================== */

struct ECPoint {
    uint256 x, y, z;

    __device__ __forceinline__ ECPoint()
        : x(UINT256_ZERO), y(UINT256_ZERO), z(UINT256_ZERO) {}

    __device__ __forceinline__ bool is_infinity() const {
        return z.is_zero();
    }

    __device__ __forceinline__ void set_infinity() {
        x = UINT256_ZERO;
        y = UINT256_ZERO;
        z = UINT256_ZERO;
    }
};

/* ==================== 生成元 G ==================== */

/*
 * 返回 secp256k1 生成元 G 的 Jacobian 坐标 (Z=1)
 *
 * 生成元仿射坐标:
 *   G.x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
 *   G.y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
 *
 * uint256 构造函数参数顺序: (v[0]=LSB, v[1], v[2], v[3]=MSB)
 */
__device__ __forceinline__ ECPoint get_generator() {
    ECPoint G;
    G.x = uint256(0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                  0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL);
    G.y = uint256(0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
                  0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL);
    G.z = UINT256_ONE;
    return G;
}

/* ==================== 点取反 ==================== */

/*
 * 返回 -P = (X, -Y mod p, Z) 在 Jacobian 坐标下
 * 仿射取反: (x, y) → (x, p - y)
 * Jacobian: (X, Y, Z) → (X, p - Y, Z)，因为 -(Y/Z³) = (p-Y)/Z³
 */
__device__ __forceinline__ ECPoint point_neg(const ECPoint& P) {
    ECPoint result;
    result.x = P.x;
    result.y = field_neg(P.y);
    result.z = P.z;
    return result;
}

/* ==================== 点倍乘 (secp256k1 a=0) ==================== */

/*
 * Jacobian 点倍乘: R = 2P
 *
 * secp256k1 曲线 a = 0，简化公式:
 *   如果 Y == 0: 返回无穷远点
 *   W  = 3 * X²  (a=0, 所以 aZ⁴ + 3X² = 3X²)
 *   S  = Y * Z
 *   B  = X * Y * S = X * Y² * Z
 *   H  = W² - 8B
 *   X' = 2 * H * S
 *   Y' = W * (4B - H) - 8 * Y² * S²
 *   Z' = 8 * S³
 */
__device__ ECPoint point_double(const ECPoint& P) {
    /* 无穷远点倍乘仍为无穷远点 */
    if (P.is_infinity()) {
        ECPoint inf;
        inf.set_infinity();
        return inf;
    }

    /* Y == 0 时，切线垂直，结果为无穷远点 */
    if (P.y.is_zero()) {
        ECPoint inf;
        inf.set_infinity();
        return inf;
    }

    /* 计算中间变量 */
    uint256 xx = field_sqr(P.x);       /* X² */
    uint256 w  = field_add(xx, xx);     /* 2X² */
    w = field_add(w, xx);               /* W = 3X² */

    uint256 s = field_mul(P.y, P.z);    /* S = Y*Z */

    uint256 y_sq = field_sqr(P.y);      /* Y² */
    uint256 b    = field_mul(P.x, y_sq); /* X*Y² */
    b = field_mul(b, P.z);              /* B = X*Y²*Z */
    /* 注意: B 也可以直接用 field_mul(P.x, field_mul(P.y, s)) 计算 */

    uint256 b4 = field_add(b, b);       /* 2B */
    b4 = field_add(b4, b4);             /* 4B */

    uint256 h = field_sqr(w);           /* W² */
    uint256 b8 = field_add(b4, b4);     /* 8B */
    h = field_sub(h, b8);              /* H = W² - 8B */

    /* 计算结果坐标 */
    uint256 hs = field_mul(h, s);       /* H*S */
    ECPoint R;
    R.x = field_add(hs, hs);            /* X' = 2*H*S */

    uint256 u = field_sub(b4, h);       /* 4B - H */
    uint256 wu = field_mul(w, u);       /* W*(4B-H) */
    uint256 y_sq_s_sq = field_mul(y_sq, field_sqr(s)); /* Y²*S² */
    uint256 eight_y_sq_s_sq = field_add(y_sq_s_sq, y_sq_s_sq); /* 2*Y²*S² */
    eight_y_sq_s_sq = field_add(eight_y_sq_s_sq, eight_y_sq_s_sq); /* 4*Y²*S² */
    eight_y_sq_s_sq = field_add(eight_y_sq_s_sq, eight_y_sq_s_sq); /* 8*Y²*S² */
    R.y = field_sub(wu, eight_y_sq_s_sq); /* Y' = W*(4B-H) - 8*Y²*S² */

    uint256 s3 = field_mul(field_sqr(s), s); /* S³ */
    R.z = field_add(s3, s3);            /* 2S³ */
    R.z = field_add(R.z, R.z);          /* 4S³ */
    R.z = field_add(R.z, R.z);          /* Z' = 8*S³ */

    return R;
}

/* ==================== Jacobian + Jacobian 点加法 ==================== */

/*
 * Jacobian 点加法: R = P + Q
 *
 * 公式 (当 P≠Q, P≠∞, Q≠∞):
 *   U1 = Y2 * Z1²        U2 = Y1 * Z2²
 *   V1 = X2 * Z1³        V2 = X1 * Z2³
 *   如果 V1 == V2:
 *     U1 == U2 → 返回 point_double(P) (同一点)
 *     U1 != U2 → 返回无穷远点 (互为逆元)
 *   U  = U1 - U2
 *   V  = V1 - V2
 *   W  = Z1 * Z2
 *   A  = U² * W - V³ - 2V² * V2
 *   X3 = V * A
 *   Y3 = U * (V² * V2 - A) - V³ * U2
 *   Z3 = V³ * W
 */
__device__ ECPoint point_add(const ECPoint& P, const ECPoint& Q) {
    /* 处理无穷远点的特殊情况 */
    if (P.is_infinity()) return Q;
    if (Q.is_infinity()) return P;

    /* 预计算 Z 的幂 */
    uint256 z1_sq = field_sqr(P.z);             /* Z1² */
    uint256 z2_sq = field_sqr(Q.z);             /* Z2² */
    uint256 z1_cb = field_mul(z1_sq, P.z);      /* Z1³ */
    uint256 z2_cb = field_mul(z2_sq, Q.z);      /* Z2³ */

    /* 计算 U1, U2, V1, V2 */
    uint256 u1 = field_mul(Q.y, z1_sq);         /* U1 = Y2 * Z1² */
    uint256 u2 = field_mul(P.y, z2_sq);         /* U2 = Y1 * Z2² */
    uint256 v1 = field_mul(Q.x, z1_cb);         /* V1 = X2 * Z1³ */
    uint256 v2 = field_mul(P.x, z2_cb);         /* V2 = X1 * Z2³ */

    /* V1 == V2: 两点 x 坐标相同 */
    if (v1.is_equal(v2)) {
        if (u1.is_equal(u2)) {
            /* 同一点，使用倍乘 */
            return point_double(P);
        }
        /* 互为逆元，返回无穷远点 */
        ECPoint inf;
        inf.set_infinity();
        return inf;
    }

    /* 通用加法公式 */
    uint256 u = field_sub(u1, u2);               /* U = U1 - U2 */
    uint256 v = field_sub(v1, v2);               /* V = V1 - V2 */
    uint256 w = field_mul(P.z, Q.z);             /* W = Z1 * Z2 */

    uint256 v_sq = field_sqr(v);                 /* V² */
    uint256 v_cb = field_mul(v_sq, v);           /* V³ */
    uint256 u_sq = field_sqr(u);                 /* U² */
    uint256 u_sq_w = field_mul(u_sq, w);         /* U² * W */
    uint256 v_sq_v2 = field_mul(v_sq, v2);       /* V² * V2 */

    /* A = U² * W - V³ - 2 * V² * V2 */
    uint256 a = field_sub(u_sq_w, v_cb);
    uint256 two_v_sq_v2 = field_add(v_sq_v2, v_sq_v2);
    a = field_sub(a, two_v_sq_v2);

    /* X3 = V * A */
    ECPoint R;
    R.x = field_mul(v, a);

    /* Y3 = U * (V² * V2 - A) - V³ * U2 */
    uint256 inner = field_sub(v_sq_v2, a);
    uint256 u_inner = field_mul(u, inner);
    uint256 v_cb_u2 = field_mul(v_cb, u2);
    R.y = field_sub(u_inner, v_cb_u2);

    /* Z3 = V³ * W */
    R.z = field_mul(v_cb, w);

    return R;
}

/* ==================== Jacobian + Affine 混合加法 ==================== */

/*
 * 混合点加法: R = P + Q，其中 Q.z == 1（Q 为仿射坐标）
 * 比通用 Jacobian 加法更高效，少几次乘法
 *
 * 当 Z2 = 1 时简化:
 *   U2 = Y1 (Z2²=1)
 *   V2 = X1 (Z2³=1)
 *   U1 = Y2 * Z1²
 *   V1 = X2 * Z1³
 */
__device__ ECPoint point_add_affine(const ECPoint& P, const ECPoint& Q) {
    /* 处理无穷远点 */
    if (P.is_infinity()) return Q;
    if (Q.is_infinity()) return P;

    /* 当 Q.z == 1 时的优化路径 */
    if (Q.z.is_equal(UINT256_ONE)) {
        /* Z2 = 1，简化计算 */
        uint256 z1_sq = field_sqr(P.z);           /* Z1² */
        uint256 z1_cb = field_mul(z1_sq, P.z);    /* Z1³ */

        uint256 u1 = field_mul(Q.y, z1_sq);       /* U1 = Y2 * Z1² */
        uint256 u2 = P.y;                          /* U2 = Y1 (Z2=1) */
        uint256 v1 = field_mul(Q.x, z1_cb);       /* V1 = X2 * Z1³ */
        uint256 v2 = P.x;                          /* V2 = X1 (Z2=1) */

        if (v1.is_equal(v2)) {
            if (u1.is_equal(u2)) {
                return point_double(P);
            }
            ECPoint inf;
            inf.set_infinity();
            return inf;
        }

        uint256 u = field_sub(u1, u2);             /* U = U1 - U2 */
        uint256 v = field_sub(v1, v2);             /* V = V1 - V2 */

        uint256 v_sq = field_sqr(v);               /* V² */
        uint256 v_cb = field_mul(v_sq, v);         /* V³ */
        uint256 u_sq = field_sqr(u);               /* U² */

        /* W = Z1 * Z2 = Z1 (因为 Z2=1) */
        uint256 u_sq_w = field_mul(u_sq, P.z);     /* U² * Z1 */
        uint256 v_sq_v2 = field_mul(v_sq, v2);     /* V² * V2 */

        /* A = U² * Z1 - V³ - 2 * V² * V2 */
        uint256 a = field_sub(u_sq_w, v_cb);
        uint256 two_v_sq_v2 = field_add(v_sq_v2, v_sq_v2);
        a = field_sub(a, two_v_sq_v2);

        ECPoint R;
        R.x = field_mul(v, a);                     /* X3 = V * A */

        uint256 inner = field_sub(v_sq_v2, a);
        uint256 u_inner = field_mul(u, inner);
        uint256 v_cb_u2 = field_mul(v_cb, u2);
        R.y = field_sub(u_inner, v_cb_u2);         /* Y3 = U*(V²V2-A) - V³U2 */

        R.z = field_mul(v_cb, P.z);                /* Z3 = V³ * Z1 (Z2=1) */

        return R;
    }

    /* Q.z != 1，回退到通用加法 */
    return point_add(P, Q);
}

/* ==================== 仿射化 (Jacobian → Affine) ==================== */

/*
 * 将 Jacobian 坐标点 P 转换为仿射坐标
 * x_affine = X * Z_inv²
 * y_affine = Y * Z_inv³
 * 结果的 z 坐标设为 1
 */
__device__ ECPoint point_reduce(const ECPoint& P) {
    if (P.is_infinity()) {
        ECPoint inf;
        inf.set_infinity();
        return inf;
    }

    uint256 z_inv  = field_inv(P.z);         /* Z⁻¹ */
    uint256 z_inv2 = field_sqr(z_inv);       /* Z⁻² */
    uint256 z_inv3 = field_mul(z_inv2, z_inv); /* Z⁻³ */

    ECPoint result;
    result.x = field_mul(P.x, z_inv2);       /* x = X / Z² */
    result.y = field_mul(P.y, z_inv3);       /* y = Y / Z³ */
    result.z = UINT256_ONE;                   /* 仿射坐标 Z = 1 */
    return result;
}

/* ==================== 点比较 ==================== */

/*
 * 比较两个 Jacobian 坐标点是否表示同一个仿射点
 * 条件: X1*Z2² == X2*Z1² 且 Y1*Z2³ == Y2*Z1³
 */
__device__ bool point_equal(const ECPoint& P, const ECPoint& Q) {
    /* 两个都是无穷远点 */
    if (P.is_infinity() && Q.is_infinity()) return true;
    /* 一个是无穷远点，另一个不是 */
    if (P.is_infinity() || Q.is_infinity()) return false;

    /* 比较 X 坐标: X1 * Z2² == X2 * Z1² */
    uint256 z1_sq = field_sqr(P.z);
    uint256 z2_sq = field_sqr(Q.z);
    uint256 lhs_x = field_mul(P.x, z2_sq);
    uint256 rhs_x = field_mul(Q.x, z1_sq);
    if (!lhs_x.is_equal(rhs_x)) return false;

    /* 比较 Y 坐标: Y1 * Z2³ == Y2 * Z1³ */
    uint256 z1_cb = field_mul(z1_sq, P.z);
    uint256 z2_cb = field_mul(z2_sq, Q.z);
    uint256 lhs_y = field_mul(P.y, z2_cb);
    uint256 rhs_y = field_mul(Q.y, z1_cb);
    return lhs_y.is_equal(rhs_y);
}

/* ==================== 从仿射坐标构造 ==================== */

/*
 * 从仿射坐标 (x, y) 创建 Jacobian 坐标点，Z = 1
 */
__device__ __forceinline__ ECPoint point_from_affine(const uint256& x, const uint256& y) {
    ECPoint P;
    P.x = x;
    P.y = y;
    P.z = UINT256_ONE;
    return P;
}
