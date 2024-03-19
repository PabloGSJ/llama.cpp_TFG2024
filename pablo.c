#include "pablo.h"
#include <stdio.h>

void print_quant_times(quant_times qt) {

    printf("%s - %dth iter:\n", pablo_prefix, pablo_nq);
    printf("%s - Total exec time:\t%f\n", pablo_prefix, qt.total_time);
    printf("%s - Time per block:\t%f\n", pablo_prefix, qt.total_time);
    printf("%s - Time to find max:\t%f\n", pablo_prefix, qt.total_time);
    printf("%s - Time per iter of j2:\t%f\n", pablo_prefix, qt.total_time);
    pablo_nq++;
}