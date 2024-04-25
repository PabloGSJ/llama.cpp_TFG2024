# stats.py
import sys
import json
from matplotlib import pyplot as plt
import numpy as np

# ------------------------------------------
# USSAGE:
#
# $> python3 ./stats mode [file]
# ------------------------------------------

# CONSTANTS:
TEST_FILE = "test.json"
STATS_FILE = "unistd-row_normal.json"
used_file = TEST_FILE

# Plot variables
X_AXIS_NORMAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
X_AXIS_ZEROES = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
X_AX = X_AXIS_ZEROES

# Quantization statistics related
NUM_TENSORS = 291
NUM_ROWS = 4096
NUM_HIST = 16


# FUNCTION DECLARATIONS

def big_hist() :
    # Load tensor statistics for quantization
    tensors = json.loads(open(used_file, "r").read())["tensors"]

    BIG_HIST = [0] * 16

    for t in tensors :

        for r in t["tensor"] :
            
            i = 0
            for h in r["row"] :
                BIG_HIST[i] += h
                i += 1


    plt.bar(X_AX, BIG_HIST)
    plt.show()
    return

def tensors_one_plot() :
    tensors = json.loads(open(used_file, "r").read())["tensors"]

    # initialize plot data storage
    tensor_bars = [ [0] * 16 for i in range(NUM_TENSORS) ]

    for i in range(len(tensor_bars)) :

        for r in tensors[i]["tensor"] :

            for j in range(len(tensor_bars[i])) :
                tensor_bars[i][j] += r["row"][j]

    # prepare the bar plot
    # calculate bar positions
    bar_width = 1 / NUM_TENSORS
    
    bar_positions = [0] * NUM_TENSORS
    bar_positions[0] = np.arange(len(X_AX))
    for i in range (1, NUM_TENSORS) :
        bar_positions[i] = bar_positions[0] + (bar_width * i)
    
    # craete bar plots:
    c = 0x4287f5
    for i in range(len(tensor_bars)) :
        plt.bar(bar_positions[i], tensor_bars[i], width=bar_width)
        c += 0.02

    plt.show()
    return

def rows_analysis() :
    tensors = json.loads(open(used_file, "r").read())["tensors"]

    # row information storage:
    densest_values = [0] * 16
    row_var = [ [0] * (NUM_ROWS) for i in range(len(tensors)) ]

    # calculate statistics on denser values per row
    for t in tensors :
        for i in range(NUM_ROWS) :
            r = t["tensor"][i]["row"]
            densest_values[r.index(max(r))] += 1 
            row_var[tensors.index(t)][i] = np.var(r)

    # print all necessary values
    #densest_values[0] = densest_values[1]
    print("max: ", densest_values.index(max(densest_values)), "-", max(densest_values))
    print("list:", densest_values)
    print("mean:", np.mean(densest_values))
    print("std: ", np.std(densest_values))
    #print("std/mean:", np.std(densest_values) / np.mean(densest_values))
    print("var: ", np.var(densest_values))
    #print("row var:")
    #print(row_var)
    
    # print plot
    plt.bar(X_AX, densest_values)
    plt.show()

    return




# Modes declaration:
MODES = {
    "big-hist" :            big_hist, 
    "tensors-one-plot" :    tensors_one_plot,
    "row-analysis" :       rows_analysis, 
}

# MAIN FUNCTION:

# check arguments
if (len(sys.argv) < 2) :
    # ERROR: incorrect number of parameters
    print("ERROR: bad parameters")
    sys.exit()

mode = sys.argv[1]

if (len(sys.argv) >= 3) :
    used_file = sys.argv[2]
    print("Using file", used_file)

# Select mode
if (mode in MODES) :
    MODES[mode]()
else :
    print("ERROR: bad parameters")
    sys.exit()