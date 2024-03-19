#pragma once

#include <time.h>

#define _MTOTAL
#define _MBLOCK
#define _MMAX
#define _MJ2

// VARIABLES
// Debugging prefix for easy grep search
const char pablo_prefix[] = "PABLO";

// Global variable to keep track of the number of quantizations completed
int pablo_nq = 0;

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