#!/bin/bash

OUTF=pablo.out
MODELF=models/llama-2-7b/q8_0.gguf

for i in 1 2 3 4 5 ; do
    # Measure perplexity
    ./perplexity -m $MODELF -f wikitext-2-raw/wiki.test.raw > pablo_tmpfile 2> pablo_tmpfile
 
    # Get and print final perplexity
    echo pablo_tmpfile | head -n 4 | cut -c 10-

done 