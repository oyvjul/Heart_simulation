#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mesh.h"

typedef struct
{
    int numtensor;
    double* coord;
    double* inputtensor;
    double* fibers;

} tensorfield;

void sparse_readtensorfiles(char* tensorfile,tensorfield* T,int skip);
void fiberstotensors(tensorfield* T);
void simple_averagetensors(cube *c,tensorfield* T);
void generate_tensor(cube *c,tensorfield* T, meshdata *m);
void generate_tensor_mpi(cube *c,tensorfield* T, meshdata *m, int start_x, int start_y, int start_z, int end_x, int end_y, int end_z);

#endif
