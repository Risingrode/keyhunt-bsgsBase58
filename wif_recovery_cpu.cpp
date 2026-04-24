#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "secp256k1/SECP256k1.h"
#include "hash/sha256.h"

extern Secp256K1 *secp;

extern "C" int verify_privkey_pubkey(const uint8_t* privkey_bytes, const uint8_t* target_pubkey, int target_pubkey_len, int compressed);

static const char base58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static int get_base58_value(char c) {
    for (int i = 0; i < 58; i++) {
        if (base58[i] == c) return i;
    }
    return 0;
}

static void double_sha256(const uint8_t *data, size_t len, uint8_t *hash) {
    uint8_t first[32];
    sha256((uint8_t*)data, len, first);
    sha256(first, 32, hash);
}

// Verify WIF checksum
static int verify_wif_checksum(const char *wif, uint8_t *privkey_out) {
    uint8_t decoded[64];
    int decoded_len = 0;
    int leading_zeros = 0;
    int len = strlen(wif);
    
    while (leading_zeros < len && wif[leading_zeros] == '1') {
        leading_zeros++;
    }
    
    for (int i = leading_zeros; i < len; i++) {
        char c = wif[i];
        int digit = get_base58_value(c);
        
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
    double_sha256(full, full_len - 4, hash);
    
    if (hash[0] == full[full_len - 4] && hash[1] == full[full_len - 3] &&
        hash[2] == full[full_len - 2] && hash[3] == full[full_len - 1]) {
        if (privkey_out) {
            for (int i = 0; i < 32; i++) privkey_out[i] = full[1 + i];
        }
        return 1;
    }
    return 0;
}

struct HashEntry {
    uint64_t x_prefix;
    uint32_t C_B;
    uint32_t packed;
};

uint32_t table_size;
uint32_t table_mask;
HashEntry* bsgs_table;

void insert_hash(uint64_t x, uint32_t c, uint32_t packed) {
    if (x == 0) x = 1;
    uint32_t idx = x & table_mask;
    while(bsgs_table[idx].x_prefix != 0) {
        idx = (idx + 1) & table_mask;
    }
    bsgs_table[idx].x_prefix = x;
    bsgs_table[idx].C_B = c;
    bsgs_table[idx].packed = packed;
}

Point T_G[4][256];
Point P_missing_B[10][58];
Point P_missing_A[32][58];

int A_pos[64];
int B_pos[64];
int num_A, num_B;

Point BasePoint;
Point Neg_G_S;
uint32_t pow58_mod32[64];
uint32_t C_K;
int global_wif_len;

int found_match = 0;
char final_wif[128];
const char* global_partial_wif;
const uint8_t* global_target_pubkey;
int global_target_pubkey_len;
int global_compressed;

Point compute_C_G(uint32_t C) {
    Point P; P.Clear();
    for(int i=0; i<4; i++) {
        uint8_t byte = (C >> (i*8)) & 0xFF;
        if (byte != 0) {
            if (P.isZero()) P = T_G[i][byte];
            else P = secp->Add(P, T_G[i][byte]);
        }
    }
    return P;
}

void dfs_B(int idx, Point sum_P, uint32_t sum_C, uint32_t packed) {
    if (idx == num_B) {
        Point C_P = compute_C_G(sum_C);
        C_P = secp->Negation(C_P);
        Point P_B;
        if (sum_P.isZero()) {
            P_B = C_P;
        } else if (C_P.isZero()) {
            P_B = sum_P;
        } else {
            P_B = secp->Add(sum_P, C_P);
        }
        if (!P_B.isZero()) P_B.Reduce();
        insert_hash(P_B.isZero() ? 1 : P_B.x.bits64[0], sum_C, packed);
        return;
    }
    for(int d=0; d<58; d++) {
        Point next_P = sum_P;
        if (d > 0) {
            if (next_P.isZero()) next_P = P_missing_B[idx][d];
            else next_P = secp->Add(next_P, P_missing_B[idx][d]);
        }
        uint32_t next_C = sum_C + (d * pow58_mod32[global_wif_len - 1 - B_pos[idx]]);
        dfs_B(idx + 1, next_P, next_C, packed | ((uint32_t)d << (6 * idx)));
    }
}

void check_full_match(uint64_t packed_A, uint32_t packed_B) {
    char candidate[128];
    strcpy(candidate, global_partial_wif);
    for(int i=0; i<num_A; i++) {
        int d = (packed_A >> (6 * i)) & 0x3F;
        candidate[A_pos[i]] = base58[d];
    }
    for(int i=0; i<num_B; i++) {
        int d = (packed_B >> (6 * i)) & 0x3F;
        candidate[B_pos[i]] = base58[d];
    }
    // printf("Testing candidate: %s\n", candidate);
    uint8_t privkey[32];
    if (verify_wif_checksum(candidate, privkey)) {
        printf("Checksum matched!\n");
        if (verify_privkey_pubkey(privkey, global_target_pubkey, global_target_pubkey_len, global_compressed)) {
            strcpy(final_wif, candidate);
            found_match = 1;
        }
    }
}

void dfs_A(int idx, Point sum_P, uint32_t sum_C, uint64_t packed) {
    if (found_match) return;
    if (idx == num_A) {
        Point C_P = compute_C_G(sum_C);
        C_P = secp->Negation(C_P);
        
        Point P_A;
        if (sum_P.isZero()) P_A = C_P;
        else if (C_P.isZero()) P_A = sum_P;
        else P_A = secp->Add(sum_P, C_P);
        
        Point Neg_PA = secp->Negation(P_A);
        
        Point Base_minus_PA;
        if (BasePoint.isZero()) Base_minus_PA = Neg_PA;
        else if (Neg_PA.isZero()) Base_minus_PA = BasePoint;
        else Base_minus_PA = secp->Add(BasePoint, Neg_PA);
        
        Point Target = Base_minus_PA;
        
        for(int carry = 0; carry <= 2; carry++) {
            if (carry > 0) {
                if (Target.isZero()) Target = Neg_G_S;
                else if (!Neg_G_S.isZero()) Target = secp->Add(Target, Neg_G_S);
            }
            
            Point T = Target;
            if (!T.isZero()) T.Reduce();
            
            uint64_t search_x = T.isZero() ? 1 : T.x.bits64[0];
            uint32_t h_idx = search_x & table_mask;
            
            while(bsgs_table[h_idx].x_prefix != 0) {
                if (bsgs_table[h_idx].x_prefix == search_x) {
                    uint32_t C_B = bsgs_table[h_idx].C_B;
                    uint64_t total_C = (uint64_t)C_K + sum_C + C_B;
                    if (total_C / 0x100000000ULL == (uint64_t)carry) {
                        check_full_match(packed, bsgs_table[h_idx].packed);
                        if (found_match) return;
                    }
                }
                h_idx = (h_idx + 1) & table_mask;
            }
        }
        return;
    }
    for(int d=0; d<58; d++) {
        Point next_P = sum_P;
        if (d > 0) {
            if (next_P.isZero()) next_P = P_missing_A[idx][d];
            else next_P = secp->Add(next_P, P_missing_A[idx][d]);
        }
        uint32_t next_C = sum_C + (d * pow58_mod32[global_wif_len - 1 - A_pos[idx]]);
        dfs_A(idx + 1, next_P, next_C, packed | ((uint64_t)d << (6 * idx)));
    }
}

extern "C" int cpu_wif_recovery(
    const char *partial_wif,
    const int *missing_positions,
    int num_missing,
    const uint8_t *target_pubkey,
    int target_pubkey_len,
    int compressed,
    char *result_wif
) {
    if (num_missing == 0) return 1;
    
    global_partial_wif = partial_wif;
    global_target_pubkey = target_pubkey;
    global_target_pubkey_len = target_pubkey_len;
    global_compressed = compressed;
    found_match = 0;
    
    int pos_copy[64];
    for(int i=0; i<num_missing; i++) pos_copy[i] = missing_positions[i];
    
    global_wif_len = strlen(partial_wif);
    for(int i=0; i<num_missing-1; i++) {
        for(int j=i+1; j<num_missing; j++) {
            if(pos_copy[i] < pos_copy[j]) {
                int tmp = pos_copy[i];
                pos_copy[i] = pos_copy[j];
                pos_copy[j] = tmp;
            }
        }
    }
    
    num_B = (num_missing > 8) ? 4 : num_missing / 2;
    if (num_B > 5) num_B = 5;
    num_A = num_missing - num_B;
    
    for(int i=0; i<num_B; i++) B_pos[i] = pos_copy[i];
    for(int i=0; i<num_A; i++) A_pos[i] = pos_copy[num_B + i];

    printf("[+] BSGS Mode CPU Recovery starting. Missing: %d chars.\n", num_missing);
    
    Point P_target;
    char pubhex[132] = {0};
    for(int i=0; i<target_pubkey_len; i++) sprintf(pubhex + i*2, "%02x", target_pubkey[i]);
    bool dummy_comp;
    if(!secp->ParsePublicKeyHex(pubhex, P_target, dummy_comp)) {
        printf("[-] Failed to parse target public key!\n");
        return 1;
    }
    
    pow58_mod32[0] = 1;
    for(int i=1; i<64; i++) pow58_mod32[i] = pow58_mod32[i-1] * 58;
    
    Int V_known(0);
    for(int i=0; i<global_wif_len; i++) {
        V_known.Mult(58);
        if (partial_wif[i] != '*' && partial_wif[i] != '?') {
            V_known.Add( get_base58_value(partial_wif[i]) );
        }
    }
    
    C_K = V_known.bits[0];
    Int vk = V_known;
    vk.bits[0] = 0;
    vk.Mod(&secp->order);
    
    Int Const_S(0x80);
    int s_shifts = compressed ? 296 : 288;
    for(int i=0; i<s_shifts; i++) {
        Const_S.Add(&Const_S);
        Const_S.Mod(&secp->order);
    }
    if (compressed) {
        Int pow32(1);
        for(int i=0; i<32; i++) { pow32.Add(&pow32); pow32.Mod(&secp->order); }
        Const_S.Add(&pow32);
        Const_S.Mod(&secp->order);
    }
    
    Int P_mult(1);
    int p_shifts = compressed ? 40 : 32;
    for(int i=0; i<p_shifts; i++) { P_mult.Add(&P_mult); P_mult.Mod(&secp->order); }
    
    Point P_scaled = secp->ScalarMultiplication(P_target, &P_mult);
    Point p1 = secp->ScalarMultiplication(secp->G, &vk);
    p1 = secp->Negation(p1);
    Point p2 = secp->ScalarMultiplication(secp->G, &Const_S);
    
    BasePoint = secp->Add(P_scaled, p1);
    BasePoint = secp->Add(BasePoint, p2);
    
    Int S_val(1);
    for(int i=0; i<32; i++) { S_val.Add(&S_val); S_val.Mod(&secp->order); }
    Point G_S = secp->ScalarMultiplication(secp->G, &S_val);
    Neg_G_S = secp->Negation(G_S);
    
    Point cur = secp->G;
    for(int i=0; i<4; i++) {
        T_G[i][0].Clear();
        for(int j=1; j<256; j++) {
            if (j == 1) T_G[i][j] = cur;
            else T_G[i][j] = secp->Add(T_G[i][j-1], cur);
        }
        Point next_cur = cur;
        for(int k=0; k<8; k++) next_cur = secp->Double(next_cur);
        cur = next_cur;
    }

    for(int i=0; i<num_B; i++) {
        Int exp(1);
        int p = global_wif_len - 1 - B_pos[i];
        for(int k=0; k<p; k++) { exp.Mult(58); exp.Mod(&secp->order); }
        Point base_p = secp->ScalarMultiplication(secp->G, &exp);
        P_missing_B[i][0].Clear();
        for(int d=1; d<58; d++) {
            if (d == 1) P_missing_B[i][d] = base_p;
            else P_missing_B[i][d] = secp->Add(P_missing_B[i][d-1], base_p);
        }
    }
    
    for(int i=0; i<num_A; i++) {
        Int exp(1);
        int p = global_wif_len - 1 - A_pos[i];
        for(int k=0; k<p; k++) { exp.Mult(58); exp.Mod(&secp->order); }
        Point base_p = secp->ScalarMultiplication(secp->G, &exp);
        P_missing_A[i][0].Clear();
        for(int d=1; d<58; d++) {
            if (d == 1) P_missing_A[i][d] = base_p;
            else P_missing_A[i][d] = secp->Add(P_missing_A[i][d-1], base_p);
        }
    }
    
    uint64_t b_combs = 1;
    for(int i=0; i<num_B; i++) b_combs *= 58;
    table_size = 2;
    while(table_size < b_combs * 2) table_size *= 2;
    table_mask = table_size - 1;
    bsgs_table = (HashEntry*)calloc(table_size, sizeof(HashEntry));
    if (!bsgs_table) {
        printf("[-] Failed to allocate %lu bytes for BSGS table\n", table_size * sizeof(HashEntry));
        return 1;
    }
    
    printf("[+] Populating BSGS table with %lu combinations (B steps)...\n", b_combs);
    Point start_P; start_P.Clear();
    dfs_B(0, start_P, 0, 0);
    
    uint64_t a_combs = 1;
    for(int i=0; i<num_A; i++) a_combs *= 58;
    printf("[+] Searching %lu combinations (A steps)...\n", a_combs);
    dfs_A(0, start_P, 0, 0);
    
    free(bsgs_table);
    
    if (found_match) {
        strcpy(result_wif, final_wif);
        printf("\n[+] SUCCESS! WIF recovered: %s\n", result_wif);
        return 0;
    }
    printf("\n[-] No valid WIF found.\n");
    return 1;
}
