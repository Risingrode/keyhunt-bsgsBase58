#include <stdio.h>
#include <string.h>

int main() {
    int missing_positions[] = { 21, 28, 33, 40, 48 };
    int num_missing = 5;
    
    int pos_copy[64];
    for(int i=0; i<num_missing; i++) pos_copy[i] = missing_positions[i];
    
    int global_wif_len = 51;
    for(int i=0; i<num_missing-1; i++) {
        for(int j=i+1; j<num_missing; j++) {
            if(pos_copy[i] < pos_copy[j]) {
                int tmp = pos_copy[i];
                pos_copy[i] = pos_copy[j];
                pos_copy[j] = tmp;
            }
        }
    }
    
    int num_B = (num_missing > 8) ? 4 : num_missing / 2;
    if (num_B > 5) num_B = 5;
    int num_A = num_missing - num_B;
    
    int B_pos[64];
    int A_pos[64];
    for(int i=0; i<num_B; i++) B_pos[i] = pos_copy[i];
    for(int i=0; i<num_A; i++) A_pos[i] = pos_copy[num_B + i];

    printf("A_pos: ");
    for(int i=0; i<num_A; i++) printf("%d ", A_pos[i]);
    printf("\n");

    printf("B_pos: ");
    for(int i=0; i<num_B; i++) printf("%d ", B_pos[i]);
    printf("\n");
    return 0;
}
