#include <stdio.h>
#include <time.h>
#include "test_sgemm.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <small|middle|big>\n", argv[0]);
        return 1;
    }
    
    printf("\n==================\n");
    printf("=SGEMM test start=\n");
    printf("==================\n\n");
    
    int sizes[] = {128, 256, 512, 768, 1024, 1536};
    int start, end;
    
    int repeat;
    if (strcmp(argv[1], "small") == 0) {
        start = 0;
        end = 2;
        repeat = 1000;
        printf("=====================\n===Small Size Test===\n=====================\n\n");
    }
    else if (strcmp(argv[1], "middle") == 0) {
        start = 2;
        end = 4;
        repeat = 20;
        printf("=====================\n==Middle Size Test==\n=====================\n\n");
    }
    else if (strcmp(argv[1], "big") == 0) {
        start = 4;
        end = 6;
        repeat = 3;
        printf("====================\n===Big Size Test===\n====================\n\n");
    }
    else if (strcmp(argv[1], "all") == 0) {
        printf("====================\n===All Size Test===\n====================\n\n'\n");
        start = 0;
        end = 6;
        repeat = 3;
    }
    
    for (int i = start; i < end; i++) {
        test_sgemm(sizes[i], repeat);
    }
    
    return 0;
}