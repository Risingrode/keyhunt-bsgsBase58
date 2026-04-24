#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "base58/libbase58.h"

int main() {
    char s_min[] = "KxFC1jmwwCoACiCAWZ3eXa96mBM6tb3TYzGmf6Yw111111111111";
    uint8_t min_bytes[128];
    size_t min_sz = 128;
    b58tobin(min_bytes, &min_sz, s_min, strlen(s_min));
    printf("min_sz: %lu\n", min_sz);
    for (int i = 0; i < min_sz; i++) {
        printf("%02x", min_bytes[i]);
    }
    printf("\nData at end:\n");
    for (int i = 128 - min_sz; i < 128; i++) {
        printf("%02x", min_bytes[i]);
    }
    printf("\n");
    return 0;
}
