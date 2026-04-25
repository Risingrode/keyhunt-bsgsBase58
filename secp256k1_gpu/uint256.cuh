#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

/*
 * uint256: 256-bit unsigned integer for GPU.
 * Storage: 4 x uint64_t in little-endian order (v[0] = least significant).
 * All operations are __device__ __forceinline__ for maximum GPU performance.
 */
struct uint256 {
    uint64_t v[4];

    /* ---- Constructors ---- */

    __host__ __device__ __forceinline__ uint256() : v{0ULL, 0ULL, 0ULL, 0ULL} {}

    __host__ __device__ __forceinline__ explicit uint256(uint64_t a)
        : v{a, 0ULL, 0ULL, 0ULL} {}

    __host__ __device__ __forceinline__ uint256(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3)
        : v{v0, v1, v2, v3} {}

    /* ---- Comparison ---- */

    __device__ __forceinline__ bool is_zero() const {
        return (v[0] | v[1] | v[2] | v[3]) == 0ULL;
    }

    __device__ __forceinline__ bool is_equal(const uint256& other) const {
        return v[0] == other.v[0] &&
               v[1] == other.v[1] &&
               v[2] == other.v[2] &&
               v[3] == other.v[3];
    }

    /* ---- Addition ---- */

    /*
     * add: this += other, returns carry (0 or 1).
     */
    __device__ __forceinline__ uint64_t add(const uint256& other) {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t sum1 = v[i] + other.v[i];
            uint64_t c1 = (sum1 < v[i]) ? 1ULL : 0ULL;
            uint64_t sum2 = sum1 + carry;
            uint64_t c2 = (sum2 < sum1) ? 1ULL : 0ULL;
            v[i] = sum2;
            carry = c1 | c2;
        }
        return carry;
    }

    /*
     * add_with_carry: result = a + b + carry_in, returns new carry.
     * Writes to this, returns carry out.
     */
    __device__ __forceinline__ uint64_t add_with_carry(
        const uint256& a, const uint256& b, uint64_t carry_in)
    {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t sum1 = a.v[i] + b.v[i];
            uint64_t c1 = (sum1 < a.v[i]) ? 1ULL : 0ULL;
            uint64_t sum2 = sum1 + carry_in;
            uint64_t c2 = (sum2 < sum1) ? 1ULL : 0ULL;
            v[i] = sum2;
            carry_in = c1 | c2;
        }
        return carry_in;
    }

    /* ---- Subtraction ---- */

    /*
     * sub: this -= other, returns borrow (0 or 1).
     */
    __device__ __forceinline__ uint64_t sub(const uint256& other) {
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t sub1 = v[i] - other.v[i];
            uint64_t br1 = (v[i] < other.v[i]) ? 1ULL : 0ULL;
            uint64_t sub2 = sub1 - borrow;
            uint64_t br2 = (sub1 < borrow) ? 1ULL : 0ULL;
            v[i] = sub2;
            borrow = br1 | br2;
        }
        return borrow;
    }

    /*
     * sub_with_borrow: result = a - b - borrow_in, returns new borrow.
     * Writes to this, returns borrow out.
     */
    __device__ __forceinline__ uint64_t sub_with_borrow(
        const uint256& a, const uint256& b, uint64_t borrow_in)
    {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t sub1 = a.v[i] - b.v[i];
            uint64_t br1 = (a.v[i] < b.v[i]) ? 1ULL : 0ULL;
            uint64_t sub2 = sub1 - borrow_in;
            uint64_t br2 = (sub1 < borrow_in) ? 1ULL : 0ULL;
            v[i] = sub2;
            borrow_in = br1 | br2;
        }
        return borrow_in;
    }

    /* ---- Negation ---- */

    /*
     * neg: returns (2^256 - x) as a uint256.
     * This is equivalent to bitwise NOT + 1.
     */
    __device__ __forceinline__ uint256 neg() const {
        uint256 result;
        uint64_t carry = 1;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t inv = ~v[i];
            uint64_t sum = inv + carry;
            carry = (sum < inv) ? 1ULL : 0ULL;
            result.v[i] = sum;
        }
        return result;
    }

    /* ---- Shift operations ---- */

    /*
     * shift_left_1: this <<= 1, returns the bit shifted out of the top.
     */
    __device__ __forceinline__ uint64_t shift_left_1() {
        uint64_t carry_out = v[3] >> 63;
        #pragma unroll
        for (int i = 3; i > 0; i--) {
            v[i] = (v[i] << 1) | (v[i - 1] >> 63);
        }
        v[0] <<= 1;
        return carry_out;
    }

    /*
     * shift_right_1: this >>= 1, returns the bit shifted out of the bottom.
     */
    __device__ __forceinline__ uint64_t shift_right_1() {
        uint64_t carry_out = v[0] & 1ULL;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            v[i] = (v[i] >> 1) | (v[i + 1] << 63);
        }
        v[3] >>= 1;
        return carry_out;
    }

    /* ---- Bit access ---- */

    __device__ __forceinline__ int get_bit(int n) const {
        int word = n >> 6;          /* n / 64 */
        int bit  = n & 63;          /* n % 64 */
        return (int)((v[word] >> bit) & 1ULL);
    }

    __device__ __forceinline__ void set_bit(int n) {
        int word = n >> 6;
        int bit  = n & 63;
        v[word] |= (1ULL << bit);
    }

    __device__ __forceinline__ void clear_bit(int n) {
        int word = n >> 6;
        int bit  = n & 63;
        v[word] &= ~(1ULL << bit);
    }

    /*
     * bit_length: returns the index of the highest set bit + 1.
     * Returns 0 if the value is zero.
     */
    __device__ __forceinline__ int bit_length() const {
        /* Check from most significant word down */
        for (int i = 3; i >= 0; i--) {
            if (v[i] != 0) {
                /* Use __clzll to find the position of the highest bit */
                int leading_zeros = __clzll(v[i]);
                return (i * 64) + (64 - leading_zeros);
            }
        }
        return 0;
    }

    /*
     * get_byte: returns byte n (0 = least significant byte).
     * n must be in [0, 31].
     */
    __device__ __forceinline__ uint8_t get_byte(int n) const {
        int word = n >> 3;          /* n / 8 */
        int byte_pos = n & 7;       /* n % 8 */
        return (uint8_t)((v[word] >> (byte_pos * 8)) & 0xFFULL);
    }
};

/* ---- Global constants ---- */

static const uint256 UINT256_ZERO = uint256(0, 0, 0, 0);
static const uint256 UINT256_ONE  = uint256(1, 0, 0, 0);
