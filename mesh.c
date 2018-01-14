#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mesh.h"
#include "math.h"

/*typedef struct
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
    double totalVolume;
    int numtet;
    int numnodes;
} meshdata;*/

/*double ***dallocate_3d(int x, int y, int z)
{
  int i, j;
  double *storage = (double*)malloc(x * y * z * sizeof(*storage));
  double *alloc = storage;
  double ***matrix;
  matrix = (double***)malloc(z * sizeof(double**));

  for (i = 0; i < z; i++)
  {
    matrix[i] = (double**)malloc(y * sizeof(**matrix));

    for (j = 0; j < y; j++)
    {
      matrix[i][j] = alloc;
      alloc += x;
    }
  }

  return matrix;
}

void dinit_3d(double*** matrix, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i < z; i++)
  {
    for(j = 0; j < y; j++)
    {
      for(k = 0; k < x; k++)
      {
        matrix[i][j][k] = 0.0;
      }
    }
  }
}*/

void readmesh(char* infile, meshdata *M)
{
    char fname[200];
    char suffix[20];
    FILE *fpdata;
    int dump;
    double in;
    int i, j;

    int err;

    strcpy(fname,infile);
    strcpy(suffix,".node");
    strcat(fname,suffix);
    fpdata = fopen(fname, "r");
    if(fpdata==NULL)
    {
        printf("\nFailure to open input file.\n");
        exit(0);
    }
    err = fscanf(fpdata, "%d", &M->numnodes);
    M->nodes = (double*)calloc(3*M->numnodes,sizeof(double));
    for(j=0;j<3;++j)
        err = fscanf(fpdata, "%d", &dump);
    for (i = 0; i < M->numnodes; i++)
    {
        err = fscanf(fpdata, "%d", &dump);
        for(j=0;j<3;++j)
        {
            err = fscanf(fpdata, "%lf", &in);
            M->nodes[i*3+j]=in;///10+0.5;  //Scaling the mesh can be done here
        }
    }
    fclose(fpdata);

    strcpy(fname,infile);
    strcpy(suffix,".ele");
    strcat(fname,suffix);
    fpdata = fopen(fname, "r");
    if(fpdata==NULL)
    {
        printf("\nFailure to open input file.\n");
        exit(0);
    }
    err = fscanf(fpdata, "%d", &M->numtet);
    M->elements = (int*)calloc(4*M->numtet,sizeof(int));
    for(j=0;j<2;++j)
        err = fscanf(fpdata, "%d", &dump);
    for (i = 0; i < M->numtet; i++)
    {
        err = fscanf(fpdata, "%d", &dump);
        for(j=0;j<4;++j)
            err = fscanf(fpdata, "%d", &M->elements[i*4+j]);
    }
    fclose(fpdata);

    strcpy(fname,infile);
    strcpy(suffix,".neigh");
    strcat(fname,suffix);
    fpdata = fopen(fname, "r");
    if(fpdata==NULL)
    {
        printf("\nFailure to open input file.\n");
        exit(0);
    }
    err = fscanf(fpdata, "%d", &dump);
    assert(dump==M->numtet);
    M->neighbours = (int*)calloc(4*M->numtet,sizeof(int));
    for(j=0;j<1;++j)
        err = fscanf(fpdata, "%d", &dump);
    for (i = 0; i < M->numtet; i++)
    {
        err = fscanf(fpdata, "%d", &dump);
        for(j=0;j<4;++j)
            err = fscanf(fpdata, "%d", &M->neighbours[i*4+j]);
    }
    fclose(fpdata);
}

/*void computeVolumes(meshdata* M)
{

    //#pragma omp parallel for reduction(+ : M->totalVolume)
    //for(int i=0; i<M->numtet; i++)
    for(int i=0; i<20; i++)
    {
        //compute indices to access nodes array
        int A = M->elements[i*4]*3;
        int B = M->elements[i*4+1]*3;
        int C = M->elements[i*4+2]*3;
        int D = M->elements[i*4+3]*3;

    }
    //printf("Volume %d %lf   tot %lf \n",INSPECT,M->volume[INSPECT],M->totalVolume);
}*/

void compute_minmax(double *x_max, double *x_min, double *y_max, double *y_min, double *z_max, double *z_min, meshdata *m)
{
  int i;
  *x_max = 0;
  *x_min = 0;
  *y_max = 0;
  *y_min = 0;
  *z_max = 0;
  *z_min = 0;

  for(i = 0; i < m->numnodes; i++)
  {
    if(m->nodes[i*3] < *x_min)
      *x_min = m->nodes[i*3];

    if(m->nodes[i*3] > *x_max)
      *x_max = m->nodes[i*3];

    if(m->nodes[i*3+1] < *y_min)
      *y_min = m->nodes[i*3+1];

    if(m->nodes[i*3+1] > *y_max)
      *y_max = m->nodes[i*3+1];

    if(m->nodes[i*3+2] < *z_min)
      *z_min = m->nodes[i*3+2];

    if(m->nodes[i*3+2] > *z_max)
      *z_max = m->nodes[i*3+2];
  }
}

void compute_volume(meshdata *M)
{
  int i;

  for(i=0; i<M->numtet; i++)
  {
    int A = M->elements[i*4]*3;
    int B = M->elements[i*4+1]*3;
    int C = M->elements[i*4+2]*3;
    int D = M->elements[i*4+3]*3;

    double xdiffA = M->nodes[A]-M->nodes[D];
    double xdiffB = M->nodes[B]-M->nodes[D];
    double xdiffC = M->nodes[C]-M->nodes[D];

    double ydiffA = M->nodes[A+1]-M->nodes[D+1];
    double ydiffB = M->nodes[B+1]-M->nodes[D+1];
    double ydiffC = M->nodes[C+1]-M->nodes[D+1];

    double zdiffA = M->nodes[A+2]-M->nodes[D+2];
    double zdiffB = M->nodes[B+2]-M->nodes[D+2];
    double zdiffC = M->nodes[C+2]-M->nodes[D+2];

        //cross product BxC

    double xcross = ydiffB*zdiffC - zdiffB*ydiffC;
    double ycross = zdiffB*xdiffC - xdiffB*zdiffC;
    double zcross = xdiffB*ydiffC - ydiffB*xdiffC;

        //dot product
    double voltmp = xdiffA*xcross+ydiffA*ycross+zdiffA*zcross;


        //volume
    double vol = fabs(voltmp/6);
    //M->volume[i] = vol;
    M->total_volume += vol;
    }
}

void calculate_centroid(meshdata *m)
{
  int i,j;
  m->centroid = (double*)calloc(3*m->numtet, sizeof(double));

  for(i = 0; i < m->numtet; i++)
  {
    int A = m->elements[i*4];
    int B = m->elements[i*4+1];
    int C = m->elements[i*4+2];
    int D = m->elements[i*4+3];

    for(j = 0; j < 3; j++)
    {
      m->centroid[i*3+j] = ((m->nodes[A*3+j]*m->nodes[B*3+j]*m->nodes[C*3+j]*m->nodes[D*3+j])*0.25);
    }

    //printf("%f \t %f \n ",m->nodes[i*3], m->nodes[i*3+1]);
  }
}

double determinant(double *a, double *b, double *c, double *d)
{
  const double constant = 1.0;
  double determinant;

  determinant = (a[0]*b[1]*c[2]*constant) + (a[0]*b[2]*constant*d[1]) + (a[0]*constant*c[1]*d[2])
        + (a[1]*b[0]*constant*d[2]) + (a[1]*b[2]*c[0]*constant) + (a[1]*constant*c[2]*d[0])
        + (a[2]*b[0]*c[1]*constant) + (a[2]*b[1]*constant*d[0]) + (a[2]*constant*c[0]*d[1])
        + (constant*b[0]*c[2]*d[1]) + (constant*b[1]*c[0]*d[2]) + (constant*b[2]*c[1]*d[0])
        - (a[0]*b[1]*constant*d[2]) - (a[0]*b[2]*c[1]*constant) - (a[0]*constant*c[2]*d[1])
        - (a[1]*b[0]*c[2]*constant) - (a[1]*b[2]*constant*d[0]) - (a[1]*constant*c[0]*d[2])
        - (a[2]*b[0]*constant*d[1]) - (a[2]*b[1]*c[0]*constant) - (a[2]*constant*c[1]*d[0])
        - (constant*b[0]*c[1]*d[2]) - (constant*b[1]*c[2]*d[0]) - (constant*b[2]*c[0]*d[1]);

  return determinant;
}

int inside(int numtet, int *elements, double *nodes, double point_x, double point_y, double point_z)
{
  int i, j;
  double a_vector[3], b_vector[3], c_vector[3], d_vector[3];
  int a, b, c, d;
  double point_vector[3];
  double det_0, det_1, det_2, det_3, det_4;
  int outside = 0;

  point_vector[0] = point_x;
  point_vector[1] = point_y;
  point_vector[2] = point_z;

  for(i = 0; i < numtet; i++)
  {
    a = elements[i*4];
    b = elements[i*4+1];
    c = elements[i*4+2];
    d = elements[i*4+3];

    for(j = 0; j < 3; j++)
    {
      a_vector[j] = nodes[a*3+j];
      b_vector[j] = nodes[b*3+j];
      c_vector[j] = nodes[c*3+j];
      d_vector[j] = nodes[d*3+j];
    }

    det_0 = determinant(a_vector, b_vector, c_vector, d_vector);
    det_1 = determinant(point_vector, b_vector, c_vector, d_vector);
    det_2 = determinant(a_vector, point_vector, c_vector, d_vector);
    det_3 = determinant(a_vector, b_vector, point_vector, d_vector);
    det_4 = determinant(a_vector, b_vector, c_vector, point_vector);

    if(det_0 != 0)
    {
      if(det_0 < 0)
      {
        if(det_1 < 0 && det_2 < 0 && det_3 < 0 && det_4 < 0)
        {
          return 1;
        }
      }

      if(det_0 > 0)
      {
        if(det_1 > 0 && det_2 > 0 && det_3 > 0 && det_4 > 0)
        {
          return 1;
        }
      }
    }

  }

  return 0;
}

void init_cube_grid(cube *c, meshdata *m)
{
  int i, j, k;
  double x_max;
  double x_min;
  double y_max;
  double y_min;
  double z_max;
  double z_min;
  double volume;
  ///usit/abel/u1/langguth/cardiacfiles_exchange/3Dheart.55
  //readmesh("/usit/abel/u1/langguth/cardiacfiles_exchange/3Dheart.55", m);
  readmesh("all/mesh_new/3Dheart.1", m);
  //readmesh("mesh_new/3Dheart.1", m);
  compute_minmax(&x_max, &x_min, &y_max, &y_min, &z_max, &z_min, m);
  compute_volume(m);
  //m->volume = (double*)calloc(m->numtet, sizeof(double));
  c->x_step = (fabs(x_max) + fabs(x_min))/((double)c->x-1);
  c->y_step = (fabs(y_max) + fabs(y_min))/((double)c->y-1);
  c->z_step = (fabs(z_max) + fabs(z_min))/((double)c->z-1);

  /*c->x_step = (fabs(x_max) + fabs(x_min))/((double)c->x);
  c->y_step = (fabs(y_max) + fabs(y_min))/(double)c->y;
  c->z_step = (fabs(z_max) + fabs(z_min))/(double)c->z;*/
  /*double step_test = (fabs(x_max) + fabs(x_min))/(c->x-2);
  for(i = 1; i <= c->x; i++)
  {
    double lol = x_min + ((i-1) - (3/2))*step_test;
    printf("%f ", lol);
  }
  printf("\n");*/

  for(i = 1; i <= c->x; i++)
  {
    c->grid_x[i] = x_min + c->x_step*(i-1);
    //printf("%f ", c->grid_x[i]);
  }
  //printf("\n");

  for(i = 1; i <= c->y; i++)
  {
    c->grid_y[i] = y_min + c->y_step*(i-1);
  }

  for(i = 1; i <= c->z; i++)
  {
    c->grid_z[i] = z_min + c->z_step*(i-1);
  }

  for(i = 1; i <= c->x-1; i++)
  {
    c->center_x[i] = c->grid_x[i]+(c->x_step/2);
    //printf("%f ", c->center_x[i]);
  }
  //printf("\n");

  for(i = 1; i <= c->y-1; i++)
  {
    c->center_y[i] = c->grid_y[i]+(c->y_step/2);
  }

  for(i = 1; i <= c->z-1; i++)
  {
    c->center_z[i] = c->grid_z[i]+(c->z_step/2);
  }

  //printf("x_max: %f \t x_min: %f \t y_max: %f \t y_min: %f \t z_max: %f \t z_min: %f \n", x_max, x_min, y_max, y_min, z_max, z_min);
  printf("Total heart volume: %lf \n", m->total_volume);
  printf("Total cube volume %lf \n", (c->x_step*c->y_step*c->z_step)*((double)(c->x*c->y*c->z)));
  printf("Average volume: %lf \n", c->x_step*c->y_step*c->z_step);
  printf("delta x: %f, delta y: %f delta z: %f \n", c->x_step, c->y_step, c->z_step);
}

void init_cube_grid_mpi(cube *c, meshdata *m, int start_x, int start_y, int start_z, int rank)
{
  int i, j, k, ii, jj, kk;
  double x_max;
  double x_min;
  double y_max;
  double y_min;
  double z_max;
  double z_min;
  readmesh("all/mesh_new/3Dheart.1", m);
  //readmesh("mesh_new/3Dheart.1", m);
  compute_minmax(&x_max, &x_min, &y_max, &y_min, &z_max, &z_min, m);
  compute_volume(m);

  c->x_step = (fabs(x_max) + fabs(x_min))/((double)c->x-1);
  c->y_step = (fabs(y_max) + fabs(y_min))/((double)c->y-1);
  c->z_step = (fabs(z_max) + fabs(z_min))/((double)c->z-1);
  //m->volume = (double*)calloc(m->numtet, sizeof(double));
  //printf("x_max: %f \t x_min: %f \t y_max: %f \t y_min: %f \t z_max: %f \t z_min: %f\n", x_max, x_min, y_max, y_min, z_max, z_min);
  //printf("x0: %d x1: %d \n", c->x0, c->x1);

  ii = start_x;
  for(i = c->x0; i <= c->x1; i++)
  {
    c->grid_x[ii] = x_min + c->x_step*(i-1);
    //printf("%f ", x_min + c->x_step*(i-1));
    ii++;
  }
  //printf("\n");

  jj = start_y;
  for(i = c->y0; i <= c->y1; i++)
  {
    c->grid_y[jj] = y_min + c->y_step*(i-1);
    jj++;
  }

  kk = start_z;
  for(i = c->z0; i <= c->z1; i++)
  {
    c->grid_z[kk] = z_min + c->z_step*(i-1);
    kk++;
  }

  ii = start_x;
  for(i = c->x0; i <= c->x1; i++)
  {
    c->center_x[ii] = c->grid_x[ii]+(c->x_step/2);
    //printf("%f ", c->center_x[i]);
    ii++;
  }

  jj = start_y;
  for(i = c->y0; i <= c->y1; i++)
  {
    c->center_y[jj] = c->grid_y[jj]+(c->y_step/2);
    //printf("%f ", c->center_x[i]);
    jj++;
  }

  kk = start_z;
  for(i = c->z0; i <= c->z1; i++)
  {
    c->center_z[kk] = c->grid_z[kk]+(c->z_step/2);
    //printf("%f ", c->center_x[i]);
    kk++;
  }
  //printf("\n");


  if(rank == 0)
  {
    printf("Total heart volume: %lf \n", m->total_volume);
    printf("Total cube volume %lf \n", (c->x_step*c->y_step*c->z_step)*((double)(c->x*c->y*c->z)));
    printf("Average volume: %lf \n", c->x_step*c->y_step*c->z_step);
    printf("delta x: %f, delta y: %f delta z: %f \n", c->x_step, c->y_step, c->z_step);
  }
}
