#include "pablo.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Configuration file variables
#define CONFIG_FILE "pablo.conf"
bool do_pablo;
enum _modes {SIMPLE_MODE, PABLO_MODE};

// Histogram size
#define NUM_TENSORS       291
#define NUM_ROWS          4096
#define NUM_HIST          256
// histogram declaration
unsigned pablo_histogram[NUM_TENSORS][NUM_ROWS][NUM_HIST] = {0};

// translation tables
#define ENCODING_OFFSET 128
int8_t encoding_table[256] = {
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

#define DECODING_OFFSET 8
int8_t decoding_table[16] = {
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

// pablo_q4_0 quantization radius
uint8_t q4_0_radius = 0;

bool is_init = false;

// global variables
int pablo_tid = 0;
int pablo_rid = 0;


// Q4_0 quantization-dequantization
void simple_q4_0_quantize_row(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int k);
void pablo_q4_0_quantize_row(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int k);

void simple_q4_0_dequantize_row(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_q4_0_dequantize_row(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

// Q8_0 quantization-dequantization
void simple_q8_0_quantize_row(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k);
void pablo_q8_0_quantize_row(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k);

void simple_q8_0_dequantize_row(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_q8_0_dequantize_row(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);



void pablo_init(void) {
    // only execute this function once
    if (is_init) 
        return;
    is_init = true;

    FILE *fp = fopen(CONFIG_FILE, "r");
    if (fp == NULL) {
        fprintf(stderr, "\nERROR: no %s file found!\n", CONFIG_FILE);
        exit(1);
    }

    int num = 0;
    fscanf(fp, "%d", &num);
    
    do_pablo = num == PABLO_MODE;

    if (do_pablo)
        fprintf(stdout, "\nPABLO: performing PABLO (de)quantization...\n");
    else
        fprintf(stdout, "\nPABLO: performing SIMPLE (de)quantization...\n");

    fclose(fp);
}

void pablo_print_all(void) {    // json format
    #ifdef _PABLO_PRINT_ALL
            
        FILE *pablo_file = fopen(PABLO_FILE_NAME, "w+");

        fprintf(stdout, "{\"pablo\":{");

        // print tensor histogram
        fprintf(stdout, "\"tensors\":[");

        for (int t = 0; t < NUM_TENSORS
    -3; t++) {
            fprintf(stderr, "PABLO: %d\n", t);
            fprintf(stdout, "{\"tensor\":[");

            unsigned int sum[NUM_HIST] = {0};

            // add all rows of the tensor
            for (int r = 0; r < NUM_ROWS; r++)  {
                for (int h = 0; h < NUM_HIST; h++) {

                    sum[h] += pablo_histogram[t][r][h];
                }
            }

            // print sumatories
            for (int h = 0; h < NUM_HIST-1; h++) {

                fprintf(stdout, "%u, ", sum[h]);
            }
            // last sumatory
            fprintf(stdout, "%u", sum[NUM_HIST-1]);

            fprintf(stdout, "]}, ");
        }
        // last tensor
        fprintf(stderr, "PABLO: %d\n", NUM_TENSORS
    -3);
        fprintf(stdout, "{\"tensor\":[");

        unsigned int sum[NUM_HIST] = {0};

        // add all rows of the tensor
        for (int r = 0; r < NUM_ROWS; r++)  {
            for (int h = 0; h < NUM_HIST; h++) {

                sum[h] += pablo_histogram[NUM_TENSORS
            -3][r][h];
            }
        }

        // print sumatories
        for (int h = 0; h < NUM_HIST-1; h++) {

            fprintf(stdout, "%u, ", sum[h]);
        }
        // last sumatory
        fprintf(stdout, "%u", sum[NUM_HIST-1]);

        fprintf(stdout, "]}");

        fprintf(stdout, "]");
        fprintf(stdout, "}}\n\n");

        fclose(pablo_file);

    #endif /* _PABLO_PRINT_ALL  */
}

// Assign functions 
void pablo_quantize_row_q4_0_assign(const float * restrict x, block_q4_0 * restrict y, int k) {
    pablo_init();

    simple_q4_0_quantize_row(x, y, k);
}

void pablo_dequantize_row_q4_0_assign(const block_q4_0 * restrict x, float * restrict y, int k) {
    pablo_init();

    simple_q4_0_dequantize_row(x, y, k);
}

void pablo_quantize_row_q8_0_assign(const float * restrict x, block_q8_0 * restrict y, int k) {
    pablo_init();

    if (do_pablo) 
        pablo_q8_0_quantize_row(x, y, k);
    else
        simple_q8_0_quantize_row(x, y, k);
    
}

void pablo_dequantize_row_q8_0_assign(const block_q8_0 * restrict x, float * restrict y, int k) {
    pablo_init();

    if (do_pablo)
        pablo_q8_0_dequantize_row(x, y, k);
    else 
        simple_q8_0_dequantize_row(x, y, k);
}



// Quantization - Dequantization functions
void simple_q4_0_quantize_row(const float * restrict x, block_q4_0 * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {  
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

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

        for (int j = 0; j < qk/2; ++j) {

            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));   // value in [0, 15]
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void pablo_q4_0_quantize_row(const float * restrict x, block_q4_0 * restrict y, int k) {

    static const int qk = QK4_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {  
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

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

        for (int j = 0; j < qk/2; ++j) {

            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            int8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)) - 8;
            int8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f)) - 8;

            // if quantized value is in rage, zero it
            if (xi0 < 0 + q4_0_radius && xi0 > 0 - q4_0_radius) 
                xi0 = 0;
            
            if (xi1 < 0 + q4_0_radius && xi1 > 0 - q4_0_radius) 
                xi1 = 0;
            

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void simple_q4_0_dequantize_row(const block_q4_0 * restrict x, float * restrict y, int k) {
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

void pablo_q4_0_dequantize_row(const block_q4_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void simple_q8_0_quantize_row(const float * restrict x, block_q8_0 * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void pablo_q8_0_quantize_row(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k) {

    // fully quantize to q8_0
    simple_q8_0_quantize_row(x, y, k);

    // translate to 16 bit values
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    int tmp;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK8_0; ++j) {
            tmp = y[i].qs[j];
            y[i].qs[j] = encoding_table[tmp + ENCODING_OFFSET];

            // fprintf(stdout, "PABLO: y[%d].qs[%d] = encoding_table[%d + %d] = %d\n", i, j, tmp, ENCODING_OFFSET, y[i].qs[j]);

        }
    }
}

void simple_q8_0_dequantize_row(const block_q8_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;

            fprintf(stdout, "PABLO: y[%d*%d + %d] = decoding_table[%d + %d] * %f = %f\n", i, QK8_0, j, x[i].qs[j], DECODING_OFFSET, d, y[i*QK8_0 + j]);
        }
    }
}

void pablo_q8_0_dequantize_row(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k) {
    static const int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*QK8_0 + j] = decoding_table[x[i].qs[j] + DECODING_OFFSET] * d; 

            fprintf(stdout, "PABLO: y[%d*%d + %d] = decoding_table[%d + %d] * %f = %f\n", i, QK8_0, j, x[i].qs[j], DECODING_OFFSET, d, y[i*QK8_0 + j]);
        }
    }
}