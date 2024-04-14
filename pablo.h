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

#define PABLO_NUM_TENSORS       300     // 291
#define PABLO_NUM_ROWS          5000    // 4096
#define PABLO_NUM_HIST          16

// VARIABLES
extern int pablo_tid;   // active tensor id
extern int pablo_rid;   // active row id

// histogram with maximum granularity
extern int pablo_histogram[PABLO_NUM_TENSORS][PABLO_NUM_ROWS][PABLO_NUM_HIST];

// Pablo lib init function
void pablo_init();

// Print the information about the different histograms
void pablo_print_row(int rid);
void pablo_print_tensor(void);
void pablo_print_all(void);

#endif /* PABLO_H */