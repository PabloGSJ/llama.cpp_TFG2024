#include "pablo.h"
#include <iostream>
#include <iomanip>

// Define pablo.h functions
void pablo_print_all(void) {
    #ifdef _PABLO_PRINT_ALL

        for (int t = 0; t < PABLO_NUM_TENSORS; t++) 
            for (int r = 0; r < PABLO_NUM_ROWS; r++) 
                for (int h = 0; h < PABLO_NUM_HIST; h++) 
                    std::cout << PABLO_PREFIX << std::setw(20) <<
                              " - Tensor " << t << 
                              " - Row " << r << 
                              " - Hist " << h << 
                              ": " << pablo_histogram[t][r][h] << std::endl;

    #endif /* _PABLO_PRINT_ALL */
}