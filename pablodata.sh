#!/bin/bash

make -j 16
rm -r $HOME/json
mkdir $HOME/json

# simple_q4_0
echo 0 0 0 > pablo.conf 
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q4_0 1 | sed -n '6p' > $HOME/json/simple_q4-0.json

# pablo_q4_0 radius=[0, 1, 2, 3]
for i in 0 1 2 3 ; do
echo 1 $i 0 > pablo.conf 
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q4_0 1 | sed -n '6p' > $HOME/json/pablo_q4-0_r-$i.json
done

# simple_q8_0
echo 0 0 0 > pablo.conf 
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q8_0 1 | sed -n '6p' > $HOME/json/simple_q8-0.json

# pablo_q8_0 table=basic
echo 1 0 0 > pablo.conf 
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q8_0 1 | sed -n '6p' > $HOME/json/pablo_q8-0_basict.json

# pablo_q8_0 table=balanced
echo 1 0 1 > pablo.conf 
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q8_0 1 | sed -n '6p' > $HOME/json/pablo_q8-0_balancedt.json
