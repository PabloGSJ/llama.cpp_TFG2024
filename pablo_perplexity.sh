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
CONF_FILE=pablo.conf

make -j 16

# Simple q4_0
out_file=simple_q4_0
echo 0 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q4_0 1

echo "Measuring $out_file..."
for i in 0 1 2 3 4 ; do
    ./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > pablo_results/$out_file-$i.stdout 2> pablo_results/$out_file-$i.stderr
done 

# Simple q8_0
out_file=simple_q8_0
echo 0 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q8_0 1

echo "Measuring $out_file..."
for i in 0 1 2 3 4 ; do
    ./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > pablo_results/$out_file-$i.stdout 2> pablo_results/$out_file-$i.stderr
done 
# Pablo q8_0
out_file=pablo_q8_0
echo 1 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q8_0 1

echo "Measuring $out_file..."
for i in 0 1 2 3 4 ; do
    ./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > pablo_results/$out_file-$i.stdout 2> pablo_results/$out_file-$i.stderr
done 