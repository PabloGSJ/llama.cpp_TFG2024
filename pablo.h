#pragma once

#ifndef PABLO_H
#define PABLO_H

#include <time.h>

#define PABLO_PREFIX    "PABLO"
#define PABLO_ROW_PREFIX        "PABLO_unistd-row"
#define PABLO_TENSOR_PREFIX     "PABLO_unistd-tensor"

#define PABLO_N_TENSORS         291
#define PABLO_TENSOR_SIDE       4096
#define PABLO_MAX_HIST_VALUE    16

// VARIABLES
extern int pablo_tid;   // active tensor id
extern int pablo_rid;   // active row id

// histogram with maximum granularity
extern int pablo_histogram[PABLO_N_TENSORS][PABLO_TENSOR_SIDE][PABLO_MAX_HIST_VALUE];

// Pablo lib init function
void pablo_init();

// Print the information about the different histograms
void pablo_print_row();
void pablo_print_tensor();

#endif /* PABLO_H */