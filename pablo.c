#include "pablo.h"
#include <stdio.h>

// Initialize pablo.h variables
int pablo_tid = 0;
int pablo_rid = 0;
int pablo_occurrences = 0;
int pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST] = {0};
int pablo_grouping_hist[PABLO_MAX_GROUPING] = {0};

// Define pablo.h functions
void pablo_print_all(void) {    // formato json
    #ifdef _PABLO_PRINT_ALL

        // print tensor histogram
        printf("{\"tensors\":[");

        for (int t = 0; t < PABLO_NUM_TENSORS; t++) {
            printf("{\"tensor\":[");

            for (int r = 0; r < PABLO_NUM_ROWS; r++)  {
                printf("{\"row\":[");

                for (int h = 0; h < PABLO_NUM_HIST; h++) {
                    printf("%d, ", pablo_histogram[t][r][h]);
                }
                printf("\b\b]}, ");
            }
            printf("\b\b]}, ");
        }
        printf("\b\b]}\n\n");

        // print grouping histogram
        printf("{\"grouping\":[");

        for (int h = 0; h < PABLO_MAX_GROUPING; h++) {
            printf("%d, ", pablo_histogram[h]);

        }
        printf("\b\b]}\n");
    #endif /* _PABLO_PRINT_ALL  */
}

void pablo_print_row() {
    #ifdef _PABLO_PRINT_ROW

        printf("%s - Tensor: %d, Row: %d\n", PABLO_ROW_PREFIX, pablo_tid, pablo_rid);   // header
        
        for (int i = 0; i < PABLO_MAX_HIST_VALUE; i++) 
            printf("%s - %d: %d\n", PABLO_ROW_PREFIX, i, pablo_histogram[pablo_tid][pablo_rid][i]);

        printf("%s\n", PABLO_ROW_PREFIX);

    #endif /* _PABLO_PRINT_ROW */
}

void pablo_print_tensor() {
    #ifdef _PABLO_PRINT_TENSOR

        printf("%s - Tensor: %d\n", PABLO_ROW_PREFIX, pablo_tid);   // header

        for (int i = 0; i < PABLO_MAX_HIST_VALUE; i++) {        
            int sum = 0;

            for (int j = 0; j < PABLO_TENSOR_SIDE; j++)
                sum += pablo_histogram[pablo_tid][j][i];
        
            printf("%s - %d: %d\n", PABLO_ROW_PREFIX, i, sum);
        }
        printf("%s\n", PABLO_ROW_PREFIX);

    #endif /* _PABLO_PRINT_TENSOR */
}