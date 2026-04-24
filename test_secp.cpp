#include <stdio.h>
#include "secp256k1/SECP256k1.h"

Secp256K1 *secp;

int main() {
    secp = new Secp256K1();
    secp->Init();
    printf("SECP initialized\n");
    return 0;
}
