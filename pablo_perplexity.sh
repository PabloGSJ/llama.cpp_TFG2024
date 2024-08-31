#!/bin/bash

MODELF=models/llama-2-7b
BASE_MODEL=ggml-model-f16.gguf
TEST_FILE=wikitext-2-raw/wiki.test.raw
CONF_FILE=pablo.conf

make -j 16

# Simple q4_0
out_file=simple-q4_0
echo 0 0 0 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q4_0 1
./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > $HOME/perplexity/$out_file.stdout 2> $HOME/perplexity/$out_file.stderr

# Simple q8_0
out_file=simple-q8_0
echo 0 0 0 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q8_0 1
./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > $HOME/perplexity/$out_file.stdout 2> $HOME/perplexity/$out_file.stderr

# Pablo q8_0 basic_table
out_file=pablo-q8_0-basict
echo 1 0 0 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q8_0 1
./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > $HOME/perplexity/$out_file.stdout 2> $HOME/perplexity/$out_file.stderr

# Pablo q8_0 balanced_table
out_file=pablo-q8_0-balancedt
echo 1 0 1 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q8_0 1
./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > $HOME/perplexity/$out_file.stdout 2> $HOME/perplexity/$out_file.stderr

# pablo q4_0, radius=[0, 1, 2, 3]
for i in 0 1 2 3 ; do

out_file=pablo-q4_0-r$i
echo 1 $i 0 > $CONF_FILE

./quantize $MODELF/$BASE_MODEL $MODELF/$out_file.gguf Q4_0 1
./perplexity -m $MODELF/$out_file.gguf -f $TEST_FILE > $HOME/perplexity/$out_file.stdout 2> $HOME/perplexity/$out_file.stderr

done
