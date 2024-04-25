#pragma once

#ifndef PABLO_H
#define PABLO_H

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

// VARIABLES
// histogram with number of times a given int is quantized to
extern int pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST];
extern int pablo_tid;   // active tensor id
extern int pablo_rid;   // active row id

// histogram with agrupation rates a given int repeats
extern int pablo_grouping_hist[PABLO_MAX_GROUPING];
extern int pablo_occurrences;



// Print the information about the different histograms
void pablo_print_row(void);
void pablo_print_tensor(void);
void pablo_print_all(void);

#endif /* PABLO_H */