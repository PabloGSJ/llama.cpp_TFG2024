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
#include <unistd.h>

// ##### VARIABLES ################################################################################################
// --- Debug 
//#define _PABLO_PRINT_ALL                  // Print all the information gathered at the end of quantization
#define _PABLO_PRECISION_QUANTIZATION        // Control which type of quantization to use


// VARIABLES
// pablo identifiers
int pablo_tid;   // active tensor id
int pablo_rid;   // active row id

extern int pablo_occurrences;   // occurrence counter for pablo_grouping_hist


// ##### FUNCTIONS ################################################################################################
// --- Auxiliary functions
void pablo_init(void);  
void pablo_print_all(void);
void pablo_update(int8_t xi0);

// --- Custom quantization functions
void pablo_quantize_row_assign(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);
void pablo_quantize_row(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);
void pablo_quantize_row_imprecise(const float * GGML_RESTRICT x, block_pablo * GGML_RESTRICT y, int k);

// --- Custom dequantization functions 
void pablo_dequantize_row_assign(const block_pablo * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_dequantize_row(const block_pablo * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void pablo_dequantize_row_imprecise(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

#endif /* PABLO_H */