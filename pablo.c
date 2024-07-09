#include "pablo.h"
#include <stdio.h>
#include <assert.h>

#define PABLO_FILE_NAME "pablo_results.json"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ##### PABLO VARIABLES ##########################################################################################
// --- Local variables and macros
#define PABLO_FILE_NAME "pablo_results.json"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// translation tables
int encoding_table[256] = {
    -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 
    -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7,
    -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6,
    -5, -5, -5, -5, -5, -5, -5, -5,
    -4, -4, -4, -4,
    -3, -3,
    -2,
    -1,
    0,
    1,
    2, 2, 
    3, 3, 3, 3, 
    4, 4, 4, 4, 4, 4, 4, 4, 
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
};
#define ENCODING_OFFSET 128

int decoding_table[16] = {
    -128,
    -64,
    -32,
    -16,
    -8,
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64
};
#define DECODING_OFFSET 8

// Histogram size
#define PABLO_NUM_TENSORS       291     // 291
#define PABLO_NUM_ROWS          4096    // 4096
#define PABLO_NUM_HIST          256
// histogram declaration
unsigned pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST] = {0};

// --- pablo.h variables
int pablo_tid = 0;
int pablo_rid = 0;

// ##### FUNCTIONS ################################################################################################
// --- Local function definitions:

// debug
void pablo_quantize_debug(const float * restrict x, block_pablo * restrict y, int k);
void pablo_dequantize_debug(const block_pablo * restrict x, float * restrict y, int k);

// --- Auxiliary functions:
/**
 * Initialize pablo data
 */
void pablo_init(void) {
    // Initialize pablo.h variables
    fprintf(stderr, "\n\nPABLO got executed!\n\n");
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

// pablo_quants
// Quantization functions
void pablo_quantize_row_assign(const float * restrict x, block_pablo * restrict y, int k) {
    //printf("PABLO: ha entrado\n");
    
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    // debug quantization
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {

            y[i].qs[j] = roundf(123);
        }
    }

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK8_0; j++) {
            
            if (y[i].qs[j] != 123) {
                printf("PABLO: Encontrada discrepancia:\n");
                printf("PABLO: y[%d].qs[%d] = %d", i, j, y[i].qs[j]);
                exit(-1);
            }
        }
    }

    
    #ifdef PABLO_PRECISION_QUANTIZATION
        //pablo_quantize_row(x, y, k);
        // quantize_row_q8_0_reference(x, y, k);
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

            //pablo_update(xi0);
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
void pablo_dequantize_row_assign(const block_pablo * restrict x, float * restrict y, int k) {
    
    // debug
    printf("PABLO: alcanzado pablo_dequantize");

    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < qk; ++j) {
            
            if (x[i].qs[j] != 123) {
                printf("PABLO: Encontrada discrepancia:\n");
                printf("PABLO: x[%d].qs[%d] = %d", i, j, x[i].qs[j]);
                exit(-1);
            }
        }
    }
    printf("PABLO: exito al comprobar!\n");
    exit(0);

    #ifdef PABLO_PRECISION_QUANTIZATION
        //pablo_dequantize_row(x, y, k);
    #endif

    #ifndef PABLO_PRECISION_QUANTIZATION
        pablo_dequantize_row_imprecise(x, y, k);
    #endif
}

void pablo_dequantize_row(const block_pablo * restrict x, float * restrict y, int k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
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