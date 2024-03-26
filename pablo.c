#include "pablo.h"
#include <stdio.h>

void print_quant_times(quant_times qt) {

    printf("%s - %dth iter: -------------------------------------------------------------\n", PABLO_PREFIX, pablo_nq);
    printf("%s - Total exec time:\t%f\n", PABLO_PREFIX, qt.total_time);
    printf("%s - Time per block:\t\t%f\n", PABLO_PREFIX, qt.total_time);
    printf("%s - Time to find max:\t%f\n", PABLO_PREFIX, qt.total_time);
    printf("%s - Time per iter of j2:\t%f\n", PABLO_PREFIX, qt.total_time);
    pablo_nq++;
}