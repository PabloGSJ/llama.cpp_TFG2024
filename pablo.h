#pragma once
#ifndef PABLO_H
#define PABLO_H

#include <time.h>

#define PABLO_PREFIX        "PABLO"
#define PABLO_N_TENSORS     291
#define PABLO_TENSOR_SIDE   4096

// VARIABLES
int pablo_histogram[PABLO_N_TENSORS][PABLO_TENSOR_SIDE][16] = {0};  // histogram with maximum granularity
int pablo_tid = 0;                                                  // active tensor id
int pablo_rid = 0;                                                  // active row id

#endif /*PABLO_H*/