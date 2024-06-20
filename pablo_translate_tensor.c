#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING 100
#define MIN(a, b) ((a) < (b) ? (a) : (b))


int main(int argc, char* argv[]) {

    if (argc < 2) {
        // ERROR: not enough arguments
        fprintf(stderr, "ERROR: not enough arguments: ./pablo_translate_tensor <filename>");
        exit(-1);
    }

    char fin_name[MAX_STRING];
    strcpy(fin_name, argv[1]);
    char *fout_name = "pablo_translated_tensor.pablo";

    unsigned char buffer[MAX_STRING];
    size_t s = MAX_STRING;

    FILE *fin = fopen(fin_name, "rb");
    FILE *fout = fopen(fout_name, "wb");

    while(s >= MAX_STRING) {
        
        s = fread(buffer, sizeof(char), MAX_STRING, fin);
        s = MIN(s, MAX_STRING);

        fwrite(buffer, sizeof(char), s, fout);
    }

}