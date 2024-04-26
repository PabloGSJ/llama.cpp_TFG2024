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

// Quantization functions
void pablo_quantize_row(const float * restrict x, block_q4_0 * restrict y, int k) {

    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    // i loop:
    for (int i = 0; i < nb; i++) {  
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        // j 1 loop:
        for (int j = 0; j < qk; j++) {

            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // j 2 loop:
        for (int j = 0; j < qk; j++) {

            const float x0 = x[i*qk + 0 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)) - 8.0f;

            y[i].qs[j]  = xi0;

            pablo_histogram[pablo_tid][pablo_rid][xi0 + 8]++;   // apply offset to save into the real values

            if (xi0 == PABLO_SEEKED_INT) {
                // keep adding occurences
                pablo_occurrences++;

            } else if (pablo_occurrences > 0) {
                // save number of occurrences observed
                if (pablo_occurrences > 16) 
                    pablo_occurrences = 16;
                pablo_grouping_hist[pablo_occurrences - 1]++;

                pablo_occurrences = 0;
            }
        }
    }
}

void pablo_quantize_row_imprecise(const float * restrict x, block_q4_0 * restrict y, int k) {

    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    // i loop:
    for (int i = 0; i < nb; i++) {  
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        // j 1 loop:
        for (int j = 0; j < qk; j++) {

            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // j 2 loop:
        for (int j = 0; j < qk/2; ++j) {

            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)) - 8.0f;
            uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f)) - 8.0f;
            if (xi0 <= 1 && xi0 >= -1)
                xi0 = 0;
            if (xi1 <= 1 && xi1 >= -1)
                xi1 = 0;

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;

            pablo_histogram[pablo_tid][pablo_rid][xi0 + 8]++;
            pablo_histogram[pablo_tid][pablo_rid][xi1 + 8]++;
        }
    }
}

// Dequantization functions
