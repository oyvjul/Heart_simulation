#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

const double SIGMA_l =  1.21321;
const double SIGMA_t =  0.2121;

void sparse_readtensorfiles(char* tensorfile,tensorfield* T,int skip)
{
    char fname[200];
    char suffix[20];
    FILE *fpdata;
    double dump;
    int err;
    int i, j;
    strcpy(fname,tensorfile);
    strcpy(suffix,".fibers");
    strcat(fname,suffix);
    fpdata = fopen(fname, "r");
    if(fpdata==NULL)
    {
        printf("\nFailure to open input file %s .\n",fname);
        exit(0);
    }
    err = fscanf(fpdata, "%d", &T->numtensor);
    int total=T->numtensor;
    T->numtensor/=skip;
    T->coord = (double*)malloc(3*T->numtensor*sizeof(double));
    T->fibers = (double*)malloc(3*T->numtensor*sizeof(double));
    int pos=0;
    for (i = 0; i < T->numtensor; i++)
    {
        if(pos<total)
        for(j=0;j<3;j++)
            err = fscanf(fpdata, "%lf", &T->fibers[i*3+j]);
        else
            break;
        for(j=0;j<3;j++)
            err = fscanf(fpdata, "%lf", &dump);

    }
    fclose(fpdata);

    strcpy(fname,tensorfile);
    strcpy(suffix,".tensorcoord");
    strcat(fname,suffix);
    fpdata = fopen(fname, "r");
    if(fpdata==NULL)
    {
        printf("\nFailure to open input file %s .\n",fname);
        exit(0);
    }
    pos=0;
    for (i = 0; i < T->numtensor; i++)
    {
        if(pos<total)
        for(j=0;j<3;j++)
            err = fscanf(fpdata, "%lf", &T->coord[i*3+j]);
        else
            break;
        for(j=0;j<3;j++)
            err = fscanf(fpdata, "%lf", &dump);
    }
    fclose(fpdata);

}



void fiberstotensors(tensorfield* T)
{
  int i, j;
    //nonzeroes on main diagonal
    // +3: (1,2) , (2,1)
    // +4: (1,3) , (3,1)
    // +5: (2,3) , (3,2)

    //double start=omp_get_wtime();
    T->inputtensor = (double*)malloc(6*T->numtensor*sizeof(double));
    for (i = 0; i < T->numtensor; i++)
    {
        for(j=0;j<3;j++)
          T->inputtensor[i*6+j]=((SIGMA_l-SIGMA_t)*T->fibers[i*3+j]*T->fibers[i*3+j])+SIGMA_t;

      T->inputtensor[i*6+3]=((SIGMA_l-SIGMA_t)*T->fibers[i*3+0]*T->fibers[i*3+1]);
      T->inputtensor[i*6+4]=((SIGMA_l-SIGMA_t)*T->fibers[i*3+0]*T->fibers[i*3+2]);
      T->inputtensor[i*6+5]=((SIGMA_l-SIGMA_t)*T->fibers[i*3+1]*T->fibers[i*3+2]);
    }
    free(T->fibers);
    //double runtime=omp_get_wtime()-start;
    //printf("Fiberstotensors. Time taken: %lf \n",runtime);
}

void simple_averagetensors(cube *c,tensorfield* T)
{
    double lambda=-1.0;  //store negative lambda
    int i, j, k, l;
    double d, e, norm;

    for(i = 1; i <= c->z+1; i++)
    {
      for(j = 1; j <= c->y+1; j++)
      {
        for(k = 1; k <= c->x+1; k++)
        {

          norm = 0.0;
          for(l = 0; l < T->numtensor; l++)
          {
            d = ((c->grid_x[i]+(c->x_step/2))-T->coord[l*3+0])*((c->grid_x[i]+(c->x_step/2))-T->coord[l*3+0])+
                ((c->grid_y[j+1])-T->coord[l*3+1])*((c->grid_y[j+1])-T->coord[l*3+1])+
                ((c->grid_z[k]+(c->z_step/2))-T->coord[l*3+2])*((c->grid_z[k]+(c->z_step/2))-T->coord[l*3+2]);

            e = exp(lambda*d);
            norm+=e;
            c->tensor_x0[i][j][k]+=T->inputtensor[6*l+0]*e;
            c->tensor_x1[i][j][k]+=T->inputtensor[6*l+1]*e;
            c->tensor_y0[i][j][k]+=T->inputtensor[6*l+2]*e;
            c->tensor_y1[i][j][k]+=T->inputtensor[6*l+3]*e;
            c->tensor_z0[i][j][k]+=T->inputtensor[6*l+4]*e;
            c->tensor_z1[i][j][k]+=T->inputtensor[6*l+5]*e;
          }

          c->tensor_x0[i][j][k]/=norm;
          c->tensor_x1[i][j][k]/=norm;
          c->tensor_y0[i][j][k]/=norm;
          c->tensor_y1[i][j][k]/=norm;
          c->tensor_z0[i][j][k]/=norm;
          c->tensor_z1[i][j][k]/=norm;
        }
      }
    }
}

void generate_tensor(cube *c,tensorfield* T, meshdata *m)
{
  double lambda=-1.0;  //store negative lambda
  int i, j, k, l;
  double d, e, norm;
  int is_inside = 0;
  for(i = 1; i <= c->nzc; i++)
  {
    for(j = 1; j <= c->nyc; j++)
    {
      for(k = 1; k <= c->nxc; k++)
      {
        is_inside = inside(m->numtet, m->elements, m->nodes, c->center_x[k], c->center_y[j], c->center_z[i]);

        if(is_inside == 1)
        {
          norm = 0.0;
          for(l = 0; l < T->numtensor; l++)
          {
            d = (c->center_x[k]-T->coord[l*3+0])*(c->center_x[k]-T->coord[l*3+0])+
                (c->center_y[j]-T->coord[l*3+1])*(c->center_y[j]-T->coord[l*3+1])+
                (c->center_z[i]-T->coord[l*3+2])*(c->center_z[i]-T->coord[l*3+2]);

            e = exp(lambda*d);
            norm += e;
            c->tensor_val11[i][j][k]+=T->inputtensor[6*l+0]*e;
            c->tensor_val12[i][j][k]+=T->inputtensor[6*l+1]*e;
            c->tensor_val13[i][j][k]+=T->inputtensor[6*l+2]*e;
            c->tensor_val22[i][j][k]+=T->inputtensor[6*l+3]*e;
            c->tensor_val23[i][j][k]+=T->inputtensor[6*l+4]*e;
            c->tensor_val33[i][j][k]+=T->inputtensor[6*l+5]*e;
          }

          c->tensor_val11[i][j][k]/=norm;
          c->tensor_val12[i][j][k]/=norm;
          c->tensor_val13[i][j][k]/=norm;
          c->tensor_val22[i][j][k]/=norm;
          c->tensor_val23[i][j][k]/=norm;
          c->tensor_val33[i][j][k]/=norm;
          //c->tensor_x0[i][j][k] = 1;
        }
        else
        {
          c->tensor_val11[i][j][k] = 0.0;
          c->tensor_val12[i][j][k] = 0.0;
          c->tensor_val13[i][j][k] = 0.0;
          c->tensor_val22[i][j][k] = 0.0;
          c->tensor_val23[i][j][k] = 0.0;
          c->tensor_val33[i][j][k] = 0.0;
        }
      }
    }
  }
}

void generate_tensor_mpi(cube *c,tensorfield* T, meshdata *m, int start_x, int start_y, int start_z, int end_x, int end_y, int end_z)
{
  double lambda=-1.0;  //store negative lambda
  int i, j, k, l;
  double d, e, norm;
  int is_inside = 0;
  for(i = start_z; i <= c->nz+end_z; i++)
  {
    for(j = start_y; j <= c->ny+end_y; j++)
    {
      for(k = start_x; k <= c->nx+end_x; k++)
      {
        is_inside = inside(m->numtet, m->elements, m->nodes, c->center_x[k], c->center_y[j], c->center_z[i]);

        if(is_inside == 1)
        {
          norm = 0.0;
          for(l = 0; l < T->numtensor; l++)
          {
            d = (c->center_x[k]-T->coord[l*3+0])*(c->center_x[k]-T->coord[l*3+0])+
                (c->center_y[j]-T->coord[l*3+1])*(c->center_y[j]-T->coord[l*3+1])+
                (c->center_z[i]-T->coord[l*3+2])*(c->center_z[i]-T->coord[l*3+2]);

            e = exp(lambda*d);
            norm += e;
            c->local_tensor_val11[i][j][k]+=T->inputtensor[6*l+0]*e;
            c->local_tensor_val12[i][j][k]+=T->inputtensor[6*l+1]*e;
            c->local_tensor_val13[i][j][k]+=T->inputtensor[6*l+2]*e;
            c->local_tensor_val22[i][j][k]+=T->inputtensor[6*l+3]*e;
            c->local_tensor_val23[i][j][k]+=T->inputtensor[6*l+4]*e;
            c->local_tensor_val33[i][j][k]+=T->inputtensor[6*l+5]*e;
          }

          c->local_tensor_val11[i][j][k]/=norm;
          c->local_tensor_val12[i][j][k]/=norm;
          c->local_tensor_val13[i][j][k]/=norm;
          c->local_tensor_val22[i][j][k]/=norm;
          c->local_tensor_val23[i][j][k]/=norm;
          c->local_tensor_val33[i][j][k]/=norm;
          //c->tensor_x0[i][j][k] = 1;
        }
        else
        {
          c->local_tensor_val11[i][j][k] = 0.0;
          c->local_tensor_val12[i][j][k] = 0.0;
          c->local_tensor_val13[i][j][k] = 0.0;
          c->local_tensor_val22[i][j][k] = 0.0;
          c->local_tensor_val23[i][j][k] = 0.0;
          c->local_tensor_val33[i][j][k] = 0.0;
        }
      }
    }
  }
}
