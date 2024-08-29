#!/bin/bash

make -j 16

# simple_q4_0
echo 0 0 0 > pablo.conf
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q4_0 1 > pablodata/tmp.json 
sed -n '6p' pablodata/tmp.json > pablodata/simple_q4_0_data.json

# pablo_q4_0 radius=0
echo 1 0 0 > pablo.conf
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q4_0 1 > pablodata/tmp.json 
sed -n '6p' pablodata/tmp.json > pablodata/pablo_r0-q4_0_data.json

# pablo_q4_0 radius=[1, 2, 3]
for i in 1 2 3 ; do
echo PABLO $i
echo 1 $i 0 > pablo.conf
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q4_0 1 > pablodata/tmp.json 
sed -n '6p' pablodata/tmp.json > pablodata/pablo_r$i-q4_0_data.json

done

# simple_q8_0
echo 0 0 0 > pablo.conf
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q8_0 1 > pablodata/tmp.json 
sed -n '6p' pablodata/tmp.json > pablodata/simple_q8_0_data.json

# pablo_q8_0 basic_table
echo 1 0 0 > pablo.conf
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q8_0 1 > pablodata/tmp.json 
sed -n '6p' pablodata/tmp.json > pablodata/pablo_basic_table-q8_0_data.json

# pablo_q8_0 balanced_table
echo 1 0 1 > pablo.conf
./quantize models/llama-2-7b/ggml-model-f16.gguf models/llama-2-7b/pablo_ignore Q8_0 1 > pablodata/tmp.json 
sed -n '6p' pablodata/tmp.json > pablodata/pablo_balanced_table-q8_0_data.json
