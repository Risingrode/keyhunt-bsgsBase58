/*
 * WIF Recovery using BSGS Algorithm on CUDA GPU
 * Header file for GPU-accelerated BSGS-based WIF private key recovery
 */

#ifndef WIF_RECOVERY_BSGS_CUDA_H
#define WIF_RECOVERY_BSGS_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU BSGS-based WIF recovery
 * 使用BSGS算法在GPU上恢复部分WIF私钥中的缺失字符
 *
 * @param partial_wif       部分WIF字符串，用'*'或'?'标记缺失字符
 * @param missing_positions 缺失位置数组（0-indexed，从左到右）
 * @param num_missing       缺失字符数量
 * @param target_pubkey     目标公钥（原始字节，33字节压缩或65字节未压缩）
 * @param target_pubkey_len 公钥字节长度（33=压缩，65=未压缩）
 * @param compressed        是否使用压缩公钥格式（影响WIF长度和偏移）
 * @param result_wif        输出缓冲区（至少64字节），成功时写入完整WIF
 * Supports up to 15 missing Base58 characters in this build:
 * up to 5 table-side chars and up to 10 search-side chars.
 *
 * @return 0=成功找到, 1=未找到, -1=无CUDA设备, -2=参数无效, -3=GPU内存分配失败, -4=GPU内核错误
 */
int cuda_wif_recovery_bsgs(
    const char* partial_wif,
    const int* missing_positions,
    int num_missing,
    const uint8_t* target_pubkey,
    int target_pubkey_len,
    int compressed,
    char* result_wif
);

#ifdef __cplusplus
}
#endif

#endif /* WIF_RECOVERY_BSGS_CUDA_H */
