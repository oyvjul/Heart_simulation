#ifndef READ_MESH_H
#define READ_MESH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
//#include "binary_cube_sequential.h"

typedef struct
{
    int* elements;
    int* neighbours;
    double* Gx;
    double* Gy;
    double* Gz;

    double* nodes;
    double* centroid;
    double* area;
    double* Nx;
    double* Ny;
    double* Nz;
    double* Sx;
    double* Sy;
    double* Sz;

    double* tensor;
    double* volume;
    double total_volume;
    int numtet;
    int numnodes;
} meshdata;

typedef struct
{
  double ***u_old;
  double ***u_new;
  double ***u_debug;
  double ***u_diff;
  double ***tensor_x0;
  double ***tensor_x1;
  double ***tensor_y0;
  double ***tensor_y1;
  double ***tensor_z0;
  double ***tensor_z1;

  double ***tensor_val11;
  double ***tensor_val12;
  double ***tensor_val13;
  double ***tensor_val22;
  double ***tensor_val23;
  double ***tensor_val33;
  double ***tensor_dummy;

  double ***local_tensor;
  double ***local_tensor_val11;
  double ***local_tensor_val12;
  double ***local_tensor_val13;
  double ***local_tensor_val22;
  double ***local_tensor_val23;
  double ***local_tensor_val33;

  double volume_cube;
  double ***gradient_x;
  double ***gradient_y;
  double ***gradient_z;

  double ***flux_x;
  double ***flux_y;
  double ***flux_z;

  double *grid_x;
  double *grid_y;
  double *grid_z;

  double *center_x;
  double *center_y;
  double *center_z;

  double x_step;
  double y_step;
  double z_step;

  int x;
  int y;
  int z;
  int nx;
  int ny;
  int nz;
  int x0, x1, y0, y1, z0, z1;
  int nxc, nyc, nzc;
}cube;

/*double ***dallocate_3d(int x, int y, int z);
void dinit_3d(double*** matrix, int x, int y, int z);*/
void readmesh(char* infile, meshdata *M);
void compute_minmax(double *x_max, double *x_min, double *y_max, double *y_min, double *z_max, double *z_min, meshdata *m);
void calculate_centroid(meshdata *m);
double determinant(double *a, double *b, double *c, double *d);
int inside(int numtet, int *elements, double *nodes, double point_x, double point_y, double point_z);
void init_cube_grid(cube *c, meshdata *m);
void init_cube_grid_mpi(cube *c, meshdata *m, int start_x, int start_y, int start_z, int rank);

#endif
