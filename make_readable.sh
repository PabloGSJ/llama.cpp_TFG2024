!/bin/bash

LLAMADIR=""

# PABLO_PREFIX ocupa los caracteres 1-5

# Muestra los id de cada uno de los tensores
cat $LLAMADIR/models/llama-2-7B/pablo.out | cut -c 1-5,25- | sort -u
echo

# Muestra los id de cada fila de cada tensor
cat $LLAMADIR/models/llama-2-7B/pablo.out | cut -c 1-5,45- | sort -u
echo

# Muestra los id de cada histograma de cada fila de cada tensor
cat $LLAMADIR/models/llama-2-7B/pablo.out | cut -c 1-5 | sort -u
echo
