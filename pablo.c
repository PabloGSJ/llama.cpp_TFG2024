#include "pablo.h"
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Initialize pablo.h variables
int pablo_tid = 0;
int pablo_rid = 0;
unsigned pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST] = {0};

long unsigned pablo_grouping_hist[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_MAX_GROUPING] = {0};
int pablo_occurrences = 0;

// initialize pablo data
void pablo_init() {

}

// out all pablo data gathered
void pablo_print_all(void) {    // formato json
    #ifdef _PABLO_PRINT_ALL
        printf("{\"pablo\":{");

        // print tensor histogram
        printf("\"tensors\":[");

        for (int t = 0; t < PABLO_NUM_TENSORS-1; t++) {
            printf("{\"tensor\":[");

            // add all rows in the tensor
            unsigned sum[PABLO_NUM_HIST] = {0};
            for (int r = 0; r < PABLO_NUM_ROWS; r++)  {

                for (int h = 0; h < PABLO_NUM_HIST; h++) {
                    sum[h] += pablo_histogram[t][r][h];
                }
            }

            for (int h = 0; h < PABLO_NUM_HIST; h++) {
                printf("%u, ", sum[h]);
            }
            printf("\b\b]}, ");
        }
/*      printf("\b\b], ");

        // print grouping histogram
        printf("\"grouping\":[");

        for (int t = 0; t < PABLO_NUM_TENSORS; t++) {
            printf("{\"tensor\":[");

            for (int r = 0; r < PABLO_NUM_ROWS; r++)  {
                printf("{\"row\":[");

                for (int h = 0; h < PABLO_NUM_HIST; h++) {
                    printf("%lu, ", pablo_grouping_hist[t][r][h]);
                }
                printf("\b\b]}, ");
            }
            printf("\b\b]}, ");
        }
        printf("\b\b],");

        // print antigrouping histogram
        printf("\"antigrouping\":[");

        for (int t = 0; t < PABLO_NUM_TENSORS; t++) {
            printf("{\"tensor\":[");

            for (int r = 0; r < PABLO_NUM_ROWS; r++)  {
                printf("{\"row\":[");

                for (int h = 0; h < PABLO_NUM_HIST; h++) {
                    printf("%lu, ", pablo_antigrouping_hist[t][r][h]);
                }
                printf("\b\b]}, ");
            }
            printf("\b\b]}, ");
        }
*/      printf("\b\b]");

        printf("}}\n\n");
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

// update pablo data
void pablo_update(int8_t xi0) {

    pablo_histogram[pablo_tid][pablo_rid][xi0 + 128]++;   // apply offset to save into the positive values

    if (xi0 == 0) {     // PABLO_SEEKED_INT
        // keep adding occurences
        pablo_occurrences++;
    } 
    else if (pablo_occurrences > 0) {
        // save number of occurrences observed
        if (pablo_occurrences > 16) 
            pablo_occurrences = 16;
        pablo_grouping_hist[pablo_tid][pablo_rid][pablo_occurrences - 1]++;
        
        fprintf(stderr, "\nPABLO gh[%d]: %lu\n", pablo_occurrences - 1, pablo_grouping_hist[pablo_tid][pablo_rid][pablo_occurrences - 1]);

        pablo_occurrences = 0;
    }
}