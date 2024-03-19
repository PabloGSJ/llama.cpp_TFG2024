#include "pablo.h"
#include <stdio.h>

void print_quant_times(quant_times qt) {

    printf("%s - %dth iter:\n", prefix, nq);
    printf("%s - Total exec time:\t%f\n", prefix, qt.total_time);
    printf("%s - Time per block:\t%f\n", prefix, qt.total_time);
    printf("%s - Time to find max:\t%f\n", prefix, qt.total_time);
    printf("%s - Time per iter of j2:\t%f\n", prefix, qt.total_time);
    nq++;
}