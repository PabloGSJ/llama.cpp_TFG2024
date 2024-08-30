#include "pablo.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// debug
#define _PABLO_MAKE_HISTS
#define _PABLO_DEBUG_OUT

// Configuration file variables
#define CONFIG_FILE "pablo.conf"
bool do_pablo;

// Histogram constants
#define NUM_TENSORS 291
#define MAX_GRP     16
#define SEEKED_INT  0
// histograms declaration
unsigned *num_hist;
int size_hist = 0;
int hist_offset = 0;
unsigned num_hist_q4_0[NUM_TENSORS * 16]  = {0}; 
unsigned num_hist_q8_0[NUM_TENSORS * 256] = {0}; 
unsigned grp_hist[MAX_GRP] = {0};
unsigned grp_occurrences = 0;

// translation tables
#define ENCODING_OFFSET 128
int8_t *encoding_table;

int8_t basic_encoding_table[256] = {
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

int8_t balanced_encoding_table[256] = {
    -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 
    -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 
    -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -7, -7, -7, -7, -7, -7, 
    -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, 
    -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -5, 
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, 
    -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, 
    -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
    4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7      
};


#define DECODING_OFFSET 8
int8_t *decoding_table;

int8_t basic_decoding_table[16] = {
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

int8_t balanced_decoding_table[16] = {-127, -65, -50, -38, -28, -19, -10, -1, 0, 9, 18, 27, 37, 49, 64, 127};

// pablo_q4_0 quantization radius
int8_t q4_0_radius = 0;

// encoding-decoding table set to use
enum _table_mode {BASIC_TABLE, BALANCED_TABLE};

// Only execute initialization function once
bool is_init = false;

// global variables
int pablo_tid = 0;
int pablo_rid = 0;


// Histogram functions
void update_hists(int value);

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

    printf("\nPABLO: Initializing data...\n");

    // inspect the configuration file
    FILE *fp = fopen(CONFIG_FILE, "r");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: no %s file found!\n", CONFIG_FILE);
        exit(1);
    }

    int mode, table;
    fscanf(fp, "%d %hhd %d", &mode, &q4_0_radius, &table);

    fclose(fp);

    // Initialize the rest of the data
    do_pablo = (mode == 1);

    if (q4_0_radius < 0) 
        q4_0_radius = 0;

    switch(table) {
        case BASIC_TABLE :
            encoding_table = basic_encoding_table;
            decoding_table = basic_decoding_table;
            break;

        case BALANCED_TABLE :
            encoding_table = balanced_encoding_table;
            decoding_table = balanced_decoding_table;
            break;
        
        default :
            fprintf(stderr, "ERROR: bad table mode in %s!\n", CONFIG_FILE);
            exit(1);
    }

    #ifdef _PABLO_DEBUG_OUT
        if (do_pablo)
            printf("PABLO:   performing PABLO (de)quantization\n");
        else 
            printf("PABLO:   performing SIMPLE (de)quantization\n");

        printf("PABLO:   using radius %u\n", q4_0_radius);

        switch(table) {
            case BASIC_TABLE :
                printf("PABLO:   using basic table set\n");
                break;
            case BALANCED_TABLE :
                printf("PABLO:   using balanced table set\n");
                break;
        }
    #endif // _PABLO_DEBUG_OUT

    is_init = true;
}

void pablo_print_all(void) {    // json format
    #ifdef _PABLO_MAKE_HISTS

        fprintf(stdout, "{\"pablo\":[");

        // print tensor histogram
        fprintf(stdout, "{\"tensors\":[");

        for (int t = 0; t < NUM_TENSORS-3; t++) {
            fprintf(stdout, "{\"hist\":[");

            for (int h = 0; h < size_hist-1; h++) {
                fprintf(stdout, "%u, ", num_hist[t*size_hist + h]);
            }
            fprintf(stdout, "%u", num_hist[t*size_hist + size_hist-1]);            // print the last hist without coma

            fprintf(stdout, "]}, ");
        }
        fprintf(stdout, "{\"hist\":[");                         // print the last tensor without coma

        for (int h = 0; h < size_hist-1; h++) {
            fprintf(stdout, "%u, ", num_hist[(NUM_TENSORS-3)*size_hist + h]);
        }
        fprintf(stdout, "%u", num_hist[(NUM_TENSORS-3)*size_hist + size_hist-1]);    // print the last hist without coma

        fprintf(stdout, "]}");

        fprintf(stdout, "]}");    // end of print tensors

        
        // print grouping histogram
        fprintf(stdout, ", {\"groups\":[");

        for (int g = 0; g < MAX_GRP-1; g++) {
            fprintf(stdout, "%u, ", grp_hist[g]);
        }
        fprintf(stdout, "%u", grp_hist[MAX_GRP-1]);                             // print the last grp without coma
        
        fprintf(stdout, "]}");  // end of print grouping

        fprintf(stdout, "]}\n");

    #endif /* _PABLO_MAKE_HISTS  */
}

void update_hists(int value) {
    #ifdef _PABLO_MAKE_HISTS

    // update num_hist
    // apply offset to account for array index
    num_hist[pablo_tid*size_hist + value + hist_offset]++;

    // update grp_hist
    if (value == SEEKED_INT) {
        // keep adding occurences
        grp_occurrences++;
    }
    else if(grp_occurrences > 0) {
        // there are stored occurences AND just found unmatching number
        // this means it's the end of a streak
        grp_hist[grp_occurrences - 1]++;
        grp_occurrences = 0;
    }

    // grp_occurrences is capped at MAX_GRP to avoid overflow
    grp_occurrences = MIN(grp_occurrences, MAX_GRP);

    #endif /* _PABLO_MAKE_HISTS */
}


// Assign functions 
void pablo_quantize_row_q4_0_assign(const float * restrict x, block_q4_0 * restrict y, int k) {
    if (!is_init) {
        num_hist = num_hist_q4_0;
        size_hist = 16;
    }
    pablo_init();

    if (do_pablo) 
        pablo_q4_0_quantize_row(x, y, k);
    else
        simple_q4_0_quantize_row(x, y, k);
}

void pablo_dequantize_row_q4_0_assign(const block_q4_0 * restrict x, float * restrict y, int k) {
    pablo_init();

    if (do_pablo) 
        pablo_q4_0_dequantize_row(x, y, k);
    else
        simple_q4_0_dequantize_row(x, y, k);
}

void pablo_quantize_row_q8_0_assign(const float * restrict x, block_q8_0 * restrict y, int k) {
    if (!is_init) {
        num_hist = num_hist_q8_0;
        size_hist = 256;
        // num_hist = num_hist_q4_0;
        // size_hist = 16;
    }
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
    hist_offset = 0;

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

            update_hists(xi0);
            update_hists(xi1);
        }
    }
}

void pablo_q4_0_quantize_row(const float * restrict x, block_q4_0 * restrict y, int k) {
    hist_offset = 8;

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
            if (xi0 <= 0 + q4_0_radius && xi0 >= 0 - q4_0_radius) 
                xi0 = 0;

            if (xi1 <= 0 + q4_0_radius && xi1 >= 0 - q4_0_radius) 
                xi1 = 0;
            

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;

            update_hists(xi0);
            update_hists(xi1);
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
    hist_offset = 128;
    
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

            update_hists(round(x0));
        }
    }
}

void pablo_q8_0_quantize_row(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k) {

    // fully quantize to q8_0
    simple_q8_0_quantize_row(x, y, k);
    hist_offset = 8;

    // translate to 16 bit values
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    int tmp;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK8_0; ++j) {
            tmp = y[i].qs[j];
            y[i].qs[j] = encoding_table[tmp + ENCODING_OFFSET];

            update_hists(encoding_table[tmp + ENCODING_OFFSET]);
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

            fprintf(stdout, "PABLO: %d * %f = %f\n", x[i].qs[j], d, y[i*qk + j]);
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
            y[i*qk + j] = decoding_table[x[i].qs[j] + DECODING_OFFSET] * d; 

            // fprintf(stdout, "PABLO: %d * %f = %f\n", decoding_table[x[i].qs[j] + DECODING_OFFSET], d, y[i*qk + j]);
        }
    }
}