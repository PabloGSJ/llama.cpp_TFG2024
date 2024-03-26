#pragma once

#include <time.h>

// DEBUG
#define _MTOTAL
#define _MBLOCK
#define _MMAX
#define _MJ2

#define PABLO_PREFIX    "PABLO"
#define N_TENSORS_7B    291         // Total number of tensors for the 7B model

// VARIABLES
int p_histogram[292][4096][16] = {0};   // histogram with maximum granularity
int p_tid = 0;                          // active tensor id
int p_rid = 0;                          // active row id

// Important measurements to print
typedef struct _pablo_measurements {
    int block_size;                 // size of the tensor subblock
    int block_ammount;              // number of blocks quantized
} pablo_measurements;

pablo_measurements pm[N_TENSORS_7B];

// id of the analized tensor
int pablo_id = 0;

// Time measurements for quantization functions
typedef struct _quant_times {   
    double total_time;          // Total executing time
    double per_block_time;      // Time it takes to quantize a block
    double max_search_time;     // Time it takes to find a max
    double j2_iter_time;        // Time it takes to complete 1 iteration of the j2 loop
} quant_times;

// FUNCTIONS
/**
 * Print the whole structure quant_times along with the id of the reading under the same prefix
 * @param qt: the structure to print
*/
void print_quant_times(quant_times qt);