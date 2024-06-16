#include "pablo.h"  

// Quantization functions
void pablo_quantize_row_assign(const float * restrict x, block_pablo * restrict y, int k) {
    
    #ifdef PABLO_PRECISION_QUANTIZATION
        pablo_quantize_row(x, y, k);
    #endif

    #ifndef PABLO_PRECISION_QUANTIZATION
        pablo_quantize_row_imprecise(x, y, k);
    #endif
}

void pablo_quantize_row(const float * restrict x, block_pablo * restrict y, int k) {

    pablo_occurrences = 0; 

    // TEMPORAL:
    int pablo_array[PABLO_NUM_HIST] = {0};
    int next_id = 0;

    // quantize_row_q8_0_reference
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

            int8_t xi0 = roundf(x0);

            y[i].qs[j] = xi0;

            pablo_update(xi0);
        }
    }
}

void pablo_quantize_row_imprecise(const float * restrict x, block_pablo * restrict y, int k) {

    pablo_occurrences = 0;
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

            int8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)) - 8;
            if (xi0 <= 2 && xi0 >= -2)
                xi0 = 0;

            y[i].qs[j]  = xi0;

            pablo_update(xi0);
        }
    }
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