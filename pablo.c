#include "pablo.h"
#include "pablo-secret.h"
#include <stdio.h>
#include <assert.h>

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

int pablo_rid = 0;
int pablo_tid = 0;

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

/**
 * Update pablo-data each loop iteration
 */
void pablo_update(int8_t xi0) {

    pablo_histogram[pablo_tid][pablo_rid][xi0 + 128]++;

    // OLD
    /*
    if (xi0 == 0) {     // PABLO_SEEKED_INT
        // keep adding occurences
        pablo_occurrences++;
    } 
    else if (pablo_occurrences > 0) {
        // save number of occurrences observed
        if (pablo_occurrences > 16) 
            pablo_occurrences = 16;
        pablo_grouping_hist[pablo_tid][pablo_rid][pablo_occurrences - 1]++;
        
        //fprintf(stderr, "\nPABLO gh[%d]: %lu\n", pablo_occurrences - 1, pablo_grouping_hist[pablo_tid][pablo_rid][pablo_occurrences - 1]);

        pablo_occurrences = 0;
    }
    */
}


// ##### Quantization functions ###################################################################################

/**
 * Select the appropriate pablo-quantization function according to the operation mode
 */
void pablo_quantize_row_assign(const float * restrict x, block_pablo * restrict y, int k) {
    
    #ifdef _PABLO_PRECISION_QUANTIZATION
        pablo_quantize_row(x, y, k);
    #endif
    #ifndef _PABLO_PRECISION_QUANTIZATION
        pablo_quantize_row_imprecise(x, y, k);
    #endif
}

/**
 * Main pablo-quantization function
 */
void pablo_quantize_row(const float * restrict x, block_pablo * restrict y, int k) {

    // fully quantize to q8_0
    quantize_row_q8_0_reference(x, y, k);

    // translate to 16 bit values
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK8_0; ++j) {
            
            y[i].qs[j] = encoding_table[y[i].qs[j] + ENCODING_OFFSET];

            pablo_update(y[i].qs[j]);
        }
    }
}

void pablo_quantize_row_imprecise(const float * restrict x, block_pablo * restrict y, int k) {
    // not implemented
}

/**
 * Debug function for various purposes
 */
void pablo_quantize_debug(const float * restrict x, block_pablo * restrict y, int k) {
    // Prepare tensor for pablo_dequantize_debug

    //printf("PABLO: Entered\n");
    
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

            y[i].qs[j] = 123;
        }
    }
}


// ##### Dequantization functions #################################################################################
/**
 * Select the appropriate pablo-dequantization function according to the operation mode
 */
void pablo_dequantize_row_assign(const block_pablo * restrict x, float * restrict y, int k) {
    
    #ifdef _PABLO_PRECISION_QUANTIZATION
        pablo_dequantize_row(x, y, k);
    #endif
    #ifndef _PABLO_PRECISION_QUANTIZATION
        pablo_dequantize_row_imprecise(x, y, k);
    #endif
}

/**
 * Main pablo-dequantization function
 */
void pablo_dequantize_row(const block_pablo * restrict x, float * restrict y, int k) {
    
    assert(k % PABLO == 0);
    const int nb = k / PABLO;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < PABLO; ++j) {

            y[i*PABLO + j] = decoding_table[x[i].qs[j] + DECODING_OFFSET];
        }
    }
}

void pablo_dequantize_row_imprecise(const block_q4_0 * restrict x, float * restrict y, int k) {
    // not implemented
}

/**
 * Debug function for various purposes
 */
void pablo_dequantize_debug(const block_pablo * restrict x, float * restrict y, int k) {
    // check that correct quantized tensors were received 

    printf("PABLO: pablo_dequantize_debug reached\n");

    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < qk; ++j) {
            
            if (x[i].qs[j] != 123) {
                printf("PABLO: Found discrepancy:\n");
                printf("PABLO: x[%d].qs[%d] = %d", i, j, x[i].qs[j]);
                exit(-1);
            }
        }
    }
    printf("PABLO: successful check!\n");
    exit(0);

}