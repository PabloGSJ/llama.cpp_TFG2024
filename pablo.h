#pragma once

#ifndef PABLO_H
#define PABLO_H

#include <time.h>

#define PABLO_PREFIX    "PABLO"

#define PABLO_N_TENSORS         291
#define PABLO_TENSOR_SIDE       4096
#define PABLO_MAX_HIST_VALUE    16

// VARIABLES
extern int pablo_tid;   // active tensor id
extern int pablo_rid;   // active row id

// histogram with maximum granularity
extern int pablo_histogram[PABLO_N_TENSORS][PABLO_TENSOR_SIDE][PABLO_MAX_HIST_VALUE];

#endif /* PABLO_H */