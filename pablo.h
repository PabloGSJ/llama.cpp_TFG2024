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

#define PABLO_MAX_GROUPING      16
#define PABLO_SEEKED_INT        0

// Control which quantization-dequantization will be used
#define PABLO_PRECISION_QUANTIZATION

// VARIABLES
// histogram with number of times a given int is quantized to
extern unsigned int pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST];
extern int pablo_tid;   // active tensor id
extern int pablo_rid;   // active row id

// histogram with agrupation rates a given int repeats
extern long unsigned pablo_grouping_hist[PABLO_MAX_GROUPING];
extern long unsigned pablo_occurrences;
extern long unsigned int pablo_unocurrences_grouping_hist[PABLO_MAX_GROUPING];
extern int pablo_unoccurrences;



// Print the information about the different histograms
void pablo_print_row(void);
void pablo_print_tensor(void);
void pablo_print_all(void);

// custom quantization and dequantization functions
void pablo_quantize_row_assign(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);
void pablo_quantize_row(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);
void pablo_quantize_row_imprecise(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);

void pablo_dequantize_row_assign(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_dequantize_row(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_dequantize_row_imprecise(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

#endif /* PABLO_H */