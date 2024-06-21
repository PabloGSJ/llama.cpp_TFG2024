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

// DEBUG
//#define _PABLO_PRINT_ROW
//#define _PABLO_PRINT_TENSOR
#define _PABLO_PRINT_ALL

#define PABLO_PREFIX    "PABLO"
#define PABLO_ROW_PREFIX        "PABLO_unistd-row"
#define PABLO_TENSOR_PREFIX     "PABLO_unistd-tensor"

#define PABLO_NUM_TENSORS       291     // 291
#define PABLO_NUM_ROWS          4096    // 4096
#define PABLO_NUM_HIST          16

// Control which quantization-dequantization will be used
#define PABLO_PRECISION_QUANTIZATION

// VARIABLES
// pablo identifiers
extern int pablo_tid;   // active tensor id
extern int pablo_rid;   // active row id

extern int pablo_occurrences;   // occurrence counter for pablo_grouping_hist

// FUNCTIONS
void pablo_init(void);


// Print pablo data
void pablo_print_all(void);
void pablo_print_row(void);
void pablo_print_tensor(void);

// Gather pablo data
void pablo_update(int8_t xi0);

// Manage pablo data

// Custom quantization and dequantization functions
void pablo_quantize_row_assign(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);
void pablo_quantize_row(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);
void pablo_quantize_row_imprecise(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);

void pablo_dequantize_row_assign(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_dequantize_row(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_dequantize_row_imprecise(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

#endif /* PABLO_H */