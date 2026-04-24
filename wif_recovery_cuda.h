/*
 * WIF Recovery using CUDA GPU
 * Header file for WIF recovery functionality
 */

#ifndef WIF_RECOVERY_CUDA_H
#define WIF_RECOVERY_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Recover missing WIF private key characters using GPU
 * 
 * @param partial_wif      Partial WIF with '*' or '?' for missing chars
 * @param missing_positions Array of positions where characters are missing
 * @param num_missing      Number of missing positions
 * @param target_pubkey    Target public key to match
 * @param target_pubkey_len Length of target public key
 * @param compressed       Whether to use compressed public keys
 * @param result_wif       Output buffer for recovered WIF (min 64 bytes)
 * @return 0 on success, 1 if not found, negative on error
 */
int cuda_wif_recovery(
    const char* partial_wif,
    const int* missing_positions,
    int num_missing,
    const uint8_t* target_pubkey,
    int target_pubkey_len,
    bool compressed,
    char* result_wif
);

#ifdef __cplusplus
}
#endif

#endif // WIF_RECOVERY_CUDA_H
