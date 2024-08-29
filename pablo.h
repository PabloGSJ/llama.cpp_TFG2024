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

#define _PABLO_PRINT_ALL

extern int pablo_tid;       // active tensor id
extern int pablo_rid;       // active row id


/**
 * Inspects a configuration file before attempting to initialize data.
 * The configuration file must have the following structure:
 * 
 * EXECUTION_MODE Q4_0_RADIUS TABLE_MODE
 * 
 * - EXECUTION_MODE: PBLO - use pablo (de)quantization function
 *                   SMPL - use the simple (de)quantization function
 * - Q4_0_RADIUS: Sets the value of the q4_0_radius variable. 
 *                This parameter is only used when executing Q4_0 in PBLO mode.
 * - TABLE_MODE: This parameter is only used when executing Q8_0 in PBLO mode.
 *               0 - BASIC_TABLE: use the basic tables for (de)quantization
 *               1 - BALANCED_TABLE: use the balanced tables for (de)quantization
 */
void pablo_init(void);

/**
 * Print all data measured during execution to stdout
 */
void pablo_print_all(void);

// Quantization - Dequantization function handlers
void pablo_quantize_row_q4_0_assign(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int k);
void pablo_dequantize_row_q4_0_assign(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

void pablo_quantize_row_q8_0_assign(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k);
void pablo_dequantize_row_q8_0_assign(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

#endif /* PABLO_H */