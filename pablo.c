#include "pablo.h"
#include <stdio.h>

int pablo_tid = 0;
int pablo_rid = 0;
int pablo_histogram[PABLO_N_TENSORS][PABLO_TENSOR_SIDE][PABLO_MAX_HIST_VALUE] = {0};


void pablo_print_row() {
    printf("%s - Tensor: %d, Row: %d\n", PABLO_ROW_PREFIX, pablo_tid, pablo_rid);

    for (int i = 0; i < PABLO_MAX_HIST_VALUE; i++) {
    
        printf("%s - %d: %d\n", PABLO_ROW_PREFIX, i, pablo_histogram[pablo_tid][pablo_rid][i]);
    }
    printf("%s\n", PABLO_ROW_PREFIX);
}

void pablo_print_tensor() {
    printf("%s - Tensor: %d\n", PABLO_ROW_PREFIX, pablo_tid);

    for (int i = 0; i < PABLO_MAX_HIST_VALUE; i++) {        
        int sum = 0;

        for (int j = 0; j < PABLO_TENSOR_SIDE; j++)
            sum += pablo_histogram[pablo_tid][j][i];
    
        printf("%s - %d: %d\n", PABLO_ROW_PREFIX, i, sum);
    }
    printf("%s\n", PABLO_ROW_PREFIX);
}