#pragma once

#ifndef PABLO_H
#define PABLO_H

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT


//#define _PABLO_PRINT_ALL  // debug

extern int pablo_tid;       // active tensor id
extern int pablo_rid;       // active row id


/**
 * 
 */
void pablo_init(void);

/**
 * Print all data measured during execution to stdout
 */
void pablo_print_all(void);

// Quantization - Dequantization function handlers
void pablo_quantize_row_q4_0_assign(const float * restrict x, block_q4_0 * restrict y, int k);
void pablo_dequantize_row_q4_0_assign(const block_q4_0 * restrict x, float * restrict y, int k);

void pablo_quantize_row_q8_0_assign(const float * restrict x, block_q8_0 * restrict y, int k);
void pablo_dequantize_row_q8_0_assign(const block_q8_0 * restrict x, float * restrict y, int k);

#endif /* PABLO_H */