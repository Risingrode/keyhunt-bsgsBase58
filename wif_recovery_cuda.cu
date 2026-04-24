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

extern "C" int verify_privkey_pubkey(const uint8_t* privkey_bytes, const uint8_t* target_pubkey, int target_pubkey_len, int compressed);

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
    const uint64_t max_combinations = 1000000000000ULL;
    for (int i = 0; i < num_missing; i++) {
        if (total_combinations > max_combinations / 58) {
            fprintf(stderr, "[E] Too many combinations. Max ~10^12\n");
            return -2;
        }
        total_combinations *= 58;
        if (total_combinations > max_combinations) {
            fprintf(stderr, "[E] Too many combinations (%" PRIu64 "). Max ~10^12\n", total_combinations);
            return -2;
        }
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
    int h_result_count = 0;
    int results_to_copy = 0;
    int match_found = 0;
    int rc = -1;
    uint64_t combinations_per_thread = 0;
    result_wif[0] = '\0';

    if (report_cuda_error(cudaMalloc((void**)&d_partial_wif, wif_len + 1), "allocate partial WIF") != 0) goto cleanup;
    if (report_cuda_error(cudaMalloc((void**)&d_missing_positions, num_missing * sizeof(int)), "allocate missing positions") != 0) goto cleanup;
    if (report_cuda_error(cudaMalloc((void**)&d_results, max_results * sizeof(WIFResult)), "allocate result buffer") != 0) goto cleanup;
    if (report_cuda_error(cudaMalloc((void**)&d_result_count, sizeof(int)), "allocate result count") != 0) goto cleanup;
    
    // Copy data to device
    if (report_cuda_error(cudaMemcpy(d_partial_wif, partial_wif, wif_len + 1, cudaMemcpyHostToDevice), "copy partial WIF") != 0) goto cleanup;
    if (report_cuda_error(cudaMemcpy(d_missing_positions, missing_positions, num_missing * sizeof(int), cudaMemcpyHostToDevice), "copy missing positions") != 0) goto cleanup;
    
    if (report_cuda_error(cudaMemcpy(d_result_count, &zero, sizeof(int), cudaMemcpyHostToDevice), "clear result count") != 0) goto cleanup;
    
    // Configure kernel launch
    combinations_per_thread = (total_combinations + (blocks * threads_per_block) - 1) / (blocks * threads_per_block);
    
    printf("[+] Launching kernel: %d blocks, %d threads\n", blocks, threads_per_block);
    printf("[+] Combinations per thread: %" PRIu64 "\n", combinations_per_thread);
    
    // Launch kernel
    wif_recovery_kernel<<<blocks, threads_per_block>>>(
        d_partial_wif,
        wif_len,
        d_missing_positions,
        num_missing,
        compressed,
        0, // start_index
        total_combinations,
        combinations_per_thread,
        d_results,
        d_result_count
    );
    
    if (report_cuda_error(cudaGetLastError(), "launch WIF recovery kernel") != 0) goto cleanup;
    if (report_cuda_error(cudaDeviceSynchronize(), "synchronize WIF recovery kernel") != 0) goto cleanup;
    
    // Check results
    if (report_cuda_error(cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost), "copy result count") != 0) goto cleanup;
    
    results_to_copy = (h_result_count > max_results) ? max_results : h_result_count;
    if (h_result_count > max_results) {
        fprintf(stderr, "[W] CUDA found %d checksum candidates; verifying the first %d\n", h_result_count, max_results);
    }
    h_results = (WIFResult*)malloc(results_to_copy * sizeof(WIFResult));
    if (results_to_copy > 0 && h_results == NULL) {
        fprintf(stderr, "[E] Failed to allocate host result buffer\n");
        goto cleanup;
    }
    
    if (results_to_copy > 0) {
        if (report_cuda_error(cudaMemcpy(h_results, d_results, results_to_copy * sizeof(WIFResult), cudaMemcpyDeviceToHost), "copy WIF candidates") != 0) {
            goto cleanup;
        }
    }

    printf("[+] GPU found %d valid WIF checksum candidates. Verifying public keys...\n", h_result_count);
    
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
