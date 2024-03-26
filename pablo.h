#pragma once
#ifndef _PABLO_H
#define _PABLO_H

#include <time.h>

// DEBUG
#define _MTOTAL
#define _MBLOCK
#define _MMAX
#define _MJ2

#define PABLO_PREFIX        "PABLO"
#define PABLO_N_TENSORS     291
#define PABLO_TENSOR_SIDE   4096

// VARIABLES
int pablo_histogram[PABLO_N_TENSORS][PABLO_TENSOR_SIDE][16] = {0};  // histogram with maximum granularity
int pablo_tid = 0;                                                  // active tensor id
int pablo_rid = 0;                                                  // active row id

#endif /*_PABLO_H*/