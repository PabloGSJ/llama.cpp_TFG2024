#include "pablo.h"
#include <stdio.h>

#define PABLO_FILE_NAME "pablo_results.json"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// pablo variables:
enum _pablo_mode {PABLO_256} pablo_mode;

// Initialize pablo.h variables
int pablo_tid = 0;
int pablo_rid = 0;
unsigned pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST] = {0};

// Local variables


// initialize pablo data
void pablo_init(void) {
    
}

// out all pablo data gathered
void pablo_print_all(void) {    // json format
    #ifdef _PABLO_PRINT_ALL
            
        FILE *pablo_file = fopen(PABLO_FILE_NAME, "w+");

        fprintf(stdout, "{\"pablo\":{");

        // print tensor histogram
        fprintf(stdout, "\"tensors\":[");

        for (int t = 0; t < PABLO_NUM_TENSORS-3; t++) {
            fprintf(stderr, "PABLO: %d\n", t);
            fprintf(stdout, "{\"tensor\":[");

            unsigned int sum[PABLO_NUM_HIST] = {0};

            // add all rows of the tensor
            for (int r = 0; r < PABLO_NUM_ROWS; r++)  {
                for (int h = 0; h < PABLO_NUM_HIST; h++) {

                    sum[h] += pablo_histogram[t][r][h];
                }
            }

            // print sumatories
            for (int h = 0; h < PABLO_NUM_HIST-1; h++) {

                fprintf(stdout, "%u, ", sum[h]);
            }
            // last sumatory
            fprintf(stdout, "%u", sum[PABLO_NUM_HIST-1]);

            fprintf(stdout, "]}, ");
        }
        // last tensor
        fprintf(stderr, "PABLO: %d\n", PABLO_NUM_TENSORS-3);
        fprintf(stdout, "{\"tensor\":[");

        unsigned int sum[PABLO_NUM_HIST] = {0};

        // add all rows of the tensor
        for (int r = 0; r < PABLO_NUM_ROWS; r++)  {
            for (int h = 0; h < PABLO_NUM_HIST; h++) {

                sum[h] += pablo_histogram[PABLO_NUM_TENSORS-3][r][h];
            }
        }

        // print sumatories
        for (int h = 0; h < PABLO_NUM_HIST-1; h++) {

            fprintf(stdout, "%u, ", sum[h]);
        }
        // last sumatory
        fprintf(stdout, "%u", sum[PABLO_NUM_HIST-1]);

        fprintf(stdout, "]}");

        fprintf(stdout, "]");
        fprintf(stdout, "}}\n\n");

        fclose(pablo_file);
    #endif /* _PABLO_PRINT_ALL  */
}

void old_pablo_print_all(void) {    // json format
    #ifdef _PABLO_PRINT_ALL
            
        FILE *pablo_file = fopen(PABLO_FILE_NAME, "w+");

        fprintf(stdout, "{\"pablo\":{");

        // print tensor histogram
        fprintf(stdout, "\"tensors\":[");

        for (int t = 0; t < PABLO_NUM_TENSORS-2; t++) {
            fprintf(stdout, "{\"tensor\":[");

            for (int r = 0; r < PABLO_NUM_ROWS-1; r++)  {
                //fprintf(stdout, "{\"row\":[");

                for (int h = 0; h < PABLO_NUM_HIST-1; h++) {

                    fprintf(stdout, "%u, ", pablo_histogram[t][r][h]);
                }
                fprintf(stdout, "%u", pablo_histogram[t][r][PABLO_NUM_HIST-1]);
                fprintf(stdout, "]}, ");
            }
            // last row
            fprintf(stdout, "{\"row\":[");

            for (int h = 0; h < PABLO_NUM_HIST-1; h++) {

                fprintf(stdout, "%u, ", pablo_histogram[t][PABLO_NUM_ROWS-1][h]);
            }
            fprintf(stdout, "%u", pablo_histogram[t][PABLO_NUM_ROWS-1][PABLO_NUM_HIST-1]);
            fprintf(stdout, "]}");

            fprintf(stdout, "]}, ");
        }
        // last tensor
        fprintf(stdout, "{\"tensor\":[");

            for (int r = 0; r < PABLO_NUM_ROWS-1; r++)  {
                fprintf(stdout, "{\"row\":[");

                for (int h = 0; h < PABLO_NUM_HIST-1; h++) {

                    fprintf(stdout, "%u, ", pablo_histogram[PABLO_NUM_TENSORS-2][r][h]);
                }
                fprintf(stdout, "%u", pablo_histogram[PABLO_NUM_TENSORS-2][r][PABLO_NUM_HIST-1]);
                fprintf(stdout, "]}, ");
            }
            // last row
            fprintf(stdout, "{\"row\":[");

            for (int h = 0; h < PABLO_NUM_HIST-1; h++) {

                fprintf(stdout, "%u, ", pablo_histogram[PABLO_NUM_TENSORS-2][PABLO_NUM_ROWS-1][h]);
            }
            fprintf(stdout, "%u", pablo_histogram[PABLO_NUM_TENSORS-2][PABLO_NUM_ROWS-1][PABLO_NUM_HIST-1]);
            fprintf(stdout, "]}");

            fprintf(stdout, "]}");

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
*/      fprintf(stdout, "]");

        fprintf(stdout, "}}\n\n");

        fclose(pablo_file);
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

    // if (xi0 == 0) {     // PABLO_SEEKED_INT
    //     // keep adding occurences
    //     pablo_occurrences++;
    // } 
    // else if (pablo_occurrences > 0) {
    //     // save number of occurrences observed
    //     if (pablo_occurrences > 16) 
    //         pablo_occurrences = 16;
    //     pablo_grouping_hist[pablo_tid][pablo_rid][pablo_occurrences - 1]++;
        
    //     //fprintf(stderr, "\nPABLO gh[%d]: %lu\n", pablo_occurrences - 1, pablo_grouping_hist[pablo_tid][pablo_rid][pablo_occurrences - 1]);

    //     pablo_occurrences = 0;
    // }
}


// pablo_quants
// Quantization functions
void pablo_quantize_row_assign(const float * restrict x, block_pablo * restrict y, int k) {
    //fprintf(stderr, "\n# PABLO: %d %u\n", pablo_tid, pablo_histogram[pablo_tid][pablo_rid][0]);
    
    #ifdef PABLO_PRECISION_QUANTIZATION
        //pablo_quantize_row(x, y, k);
        quantize_row_q8_0_reference(x, y, k);
    #endif

    #ifndef PABLO_PRECISION_QUANTIZATION
        pablo_quantize_row_imprecise(x, y, k);
    #endif
}

void pablo_quantize_row(const float * restrict x, block_pablo * restrict y, int k) {

    quantize_row_q8_0_reference(x, y, k);

    // encode
    int encoding_table[16] = {
        -65,
        -33,
        -17,
        -9,
        -5,
        -3,
        -2,
        -1,
        0,
        1,
        3,
        7,
        15,
        31,
        63,
        127
    };

    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK8_0; ++j) {

            int8_t xi0;
            for (int p = 0; p < 16; p++) {
                
                if (y[i].qs[j] <= encoding_table[p]) {
                    xi0 = p - 8;
                }
            }

            y[i].qs[j] = xi0;

            pablo_update(xi0);
        }
    }
}

void pablo_quantize_row_imprecise(const float * restrict x, block_pablo * restrict y, int k) {

//     pablo_occurrences = 0;
//     static const int qk = QK4_0;

//     assert(k % qk == 0);

//     const int nb = k / qk;

//     // i loop:
//     for (int i = 0; i < nb; i++) {  
//         float amax = 0.0f; // absolute max
//         float max  = 0.0f;

//         // j 1 loop:
//         for (int j = 0; j < qk; j++) {

//             const float v = x[i*qk + j];
//             if (amax < fabsf(v)) {
//                 amax = fabsf(v);
//                 max  = v;
//             }
//         }

//         const float d  = max / -8;
//         const float id = d ? 1.0f/d : 0.0f;

//         y[i].d = GGML_FP32_TO_FP16(d);

//         // j 2 loop:
//         for (int j = 0; j < qk; j++) {

//             const float x0 = x[i*qk + 0 + j]*id;

//             int8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)) - 8;
//             if (xi0 <= 2 && xi0 >= -2)
//                 xi0 = 0;

//             y[i].qs[j]  = xi0;

//             pablo_update(xi0);
//         }
//     }
}

// Dequantization functions
void pablo_dequantize_row_assign(const block_q4_0 * restrict x, float * restrict y, int k) {
    
    #ifdef PABLO_PRECISION_QUANTIZATION
        pablo_dequantize_row(x, y, k);
    #endif

    #ifndef PABLO_PRECISION_QUANTIZATION
        pablo_dequantize_row_imprecise(x, y, k);
    #endif
}

void pablo_dequantize_row(const block_q4_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void pablo_dequantize_row_imprecise(const block_q4_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}