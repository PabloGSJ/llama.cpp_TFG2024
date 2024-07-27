#!/bin/bash

# for i in 1 2 3 4 5 ; do
#     # Measure perplexity
#     ./perplexity -m $MODELF -f wikitext-2-raw/wiki.test.raw > pablo_tmpfile 2> pablo_tmpfile
 
#     # Get and print final perplexity
#     echo pablo_tmpfile | head -n 4 | cut -c 10-

# done 

MODELF=models/llama-2-7b
BASE_MODEL=ggml-model-f16.gguf
TEST_FILE=wikitext-2-raw/wiki.test.raw

make -j 16

# Simple q4_0
out_file=simple_q4_0

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q4_0 1
./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > $out_file.stdout 2> $out_file.stderr

# Simple q8_0

# Pablo q8_0