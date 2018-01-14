#include <stdio.h>
#include "tensor.h"
#include "mesh.h"
#include "io.h"
#include "diffusion.h"
#include "omp.h"
#include "math.h"

double ***dallocate_3d(int x, int y, int z);
void dinit_3d(double*** matrix, int x, int y, int z);
void init_sequential_data(cube *c, int x, int y, int z);
void enforce_grid_left_right(double ***E, int x, int y, int z);
void enforce_grid_up_down(double ***E, int x, int y, int z);
void enforce_grid_zup_zdown(double ***E, int x, int y, int z);
double initfunction(int i, int j, int k, cube *c);
double norm(double u)
{
  return sqrt(fabs(u*u));
}
int min(int x, int y)
{
  if(x < y)
    return x;
  else
    return y;

  return 0;
}
double min_double(double x, double y)
{
  if(x < y)
    return x;
  else
    return y;

  return 0;
}
double compute_dt(cube *c);

int main(int argc, char *argv[])
{
  /*number of points in x, y & z dimension*/
  int xx = 9;
  int yy = 9;
  int zz = 9;
  /****************************************/
  int x = xx+1;
  int y = yy+1;
  int z = zz+1;
  double count = 1.0;
  int i, j, k;
  int count_inside, count_outside;
  int count_inside_tensor, count_outside_tensor;
  double l2_diff = 0;
  double l2_norm_old = 0;
  double l2_norm_debug = 0;
  double diffusion_x, diffusion_y, diffusion_z;
  cube *c = (cube*)malloc(sizeof(cube));
  meshdata *m = (meshdata*)malloc(sizeof(meshdata));
  tensorfield *t = (tensorfield *)malloc(sizeof(tensorfield));
  double upper_right_one_x, upper_right_one_y, upper_right_one_z, lower_right_one_x,
  lower_right_one_y, lower_right_one_z, upper_right_x, upper_right_y, upper_right_z, lower_right_x,
  lower_right_y, lower_right_z, upper_left_one_x, upper_left_one_y, upper_left_one_z, lower_left_one_x,
  lower_left_one_y, lower_left_one_z, upper_left_x, upper_left_y, upper_left_z, lower_left_x, lower_left_y, lower_left_z;

  double upper_right_one, lower_right_one, upper_right, lower_right, upper_left_one, lower_left_one, upper_left, lower_left;

  init_sequential_data(c, x, y, z);
  init_cube_grid(c, m);

    count_inside_tensor = 0;
    count_outside_tensor = 0;
    count_inside = 1;
    double start3 = omp_get_wtime();
    sparse_readtensorfiles("all/mesh_new/3Dheart.1", t, 1000);
    //sparse_readtensorfiles("mesh_new/3Dheart.1", t, 1000);
    fiberstotensors(t);
    generate_tensor(c, t, m);
    printf("GENERATE_TENSORS_ONLY\n");
    double end3 = omp_get_wtime();
    printf("it took : %0.12f \n", end3-start3);

    for(i = 1; i <= c->nzc; i++)
    {
      for(j = 1; j <= c->nyc; j++)
      {
        for(k = 1; k <= c->nxc; k++)
        {
          if(c->tensor_val11[i][j][k] == 0.0)
          {
            count_outside_tensor++;
          }
          else
          {
            count_inside_tensor++;
          }
        }
      }
    }

    printf("TENSOR: inside points: %d \t outside points: %d total: %d \n", count_inside_tensor, count_outside_tensor, count_inside_tensor+count_outside_tensor);
    write_binaryformat("all/mesh_new/tensor_val11_16.tensor", c->tensor_val11, c->nxc+2, c->nyc+2, c->nzc+2);
    write_binaryformat("all/mesh_new/tensor_val12_16.tensor", c->tensor_val12, c->nxc+2, c->nyc+2, c->nzc+2);
    write_binaryformat("all/mesh_new/tensor_val13_16.tensor", c->tensor_val13, c->nxc+2, c->nyc+2, c->nzc+2);
    write_binaryformat("all/mesh_new/tensor_val22_16.tensor", c->tensor_val22, c->nxc+2, c->nyc+2, c->nzc+2);
    write_binaryformat("all/mesh_new/tensor_val23_16.tensor", c->tensor_val23, c->nxc+2, c->nyc+2, c->nzc+2);
    write_binaryformat("all/mesh_new/tensor_val33_16.tensor", c->tensor_val33, c->nxc+2, c->nyc+2, c->nzc+2);

  return 0;
}

double initfunction(int i, int j, int k, cube *c)
{
  //printf("%f \n",c->center_z[k]);
  if(c->center_x[k] >= -2.0 && c->center_x[k] <= -1.0)
  {
    //printf("yes \n");
    if(c->center_y[j] <= 3.0 && c->center_y[j] >= 1.0)
    {
      //return -85;
      //printf("yes \n");
      if(c->center_z[i] <= 1.0 && c->center_z[i] >= 0.00)
      {
        //printf("no \n");
        return -85;
      }
    }
  }
}

double ***dallocate_3d(int x, int y, int z)
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
}

void init_sequential_data(cube *c, int x, int y, int z)
{
  c->x = x;
  c->y = y;
  c->z = z;
  c->nxc = x-1;
  c->nyc = y-1;
  c->nzc = z-1;
  c->u_old = dallocate_3d(x+2, y+2, z+2);
  dinit_3d(c->u_old, x+2, y+2, z+2);

  c->u_new = dallocate_3d(x+2, y+2, z+2);
  dinit_3d(c->u_new, x+2, y+2, z+2);

  c->gradient_x = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->gradient_x, x+3, y+3, z+3);

  c->gradient_y = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->gradient_y, x+3, y+3, z+3);

  c->gradient_z = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->gradient_z, x+3, y+3, z+3);

  c->tensor_val11 = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->tensor_val11, c->nxc+2, c->nyc+2, c->nzc+2);

  c->tensor_val12 = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->tensor_val12, c->nxc+2, c->nyc+2, c->nzc+2);

  c->tensor_val13 = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->tensor_val13, c->nxc+2, c->nyc+2, c->nzc+2);

  c->tensor_val22 = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->tensor_val22, c->nxc+2, c->nyc+2, c->nzc+2);

  c->tensor_val23 = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->tensor_val23, c->nxc+2, c->nyc+2, c->nzc+2);

  c->tensor_val33 = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->tensor_val33, c->nxc+2, c->nyc+2, c->nzc+2);

  c->flux_x = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->flux_x, c->nxc+2, c->nyc+2, c->nzc+2);

  c->flux_y = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->flux_y, c->nxc+2, c->nyc+2, c->nzc+2);

  c->flux_z = dallocate_3d(c->nxc+2, c->nyc+2, c->nzc+2);
  dinit_3d(c->flux_z, c->nxc+2, c->nyc+2, c->nzc+2);

  c->grid_x = (double*)calloc(x+2, sizeof(double));
  c->grid_y = (double*)calloc(y+2, sizeof(double));
  c->grid_z = (double*)calloc(z+2, sizeof(double));

  c->center_x = (double*)calloc(x+2, sizeof(double));
  c->center_y = (double*)calloc(y+2, sizeof(double));
  c->center_z = (double*)calloc(z+2, sizeof(double));
}

double compute_dt(cube *c)
{
  int i, j, k;
  double dt;
  double tensor_max[6];
  double max_tensor_value = 0;
  double step_value = 0;
  double min_step_value = 0;

  for(i = 0; i < 6; i++)
    tensor_max[i] = 0;

  for(i = 1; i <= c->z; i++)
  {
    for(j = 1; j <= c->y; j++)
    {
      for(k = 1; k <= c->x; k++)
      {
        if(c->tensor_val11[i][j][k] > tensor_max[0])
          tensor_max[0] = c->tensor_val11[i][j][k];

        if(c->tensor_val12[i][j][k] > tensor_max[1])
          tensor_max[1] = c->tensor_val12[i][j][k];

        if(c->tensor_val13[i][j][k] > tensor_max[2])
          tensor_max[2] = c->tensor_val13[i][j][k];

        if(c->tensor_val22[i][j][k] > tensor_max[3])
          tensor_max[3] = c->tensor_val22[i][j][k];

        if(c->tensor_val23[i][j][k] > tensor_max[4])
          tensor_max[4] = c->tensor_val23[i][j][k];

        if(c->tensor_val33[i][j][k] > tensor_max[5])
          tensor_max[5] = c->tensor_val33[i][j][k];
      }
    }
  }

  for(i = 0; i < 6; i++)
  {
    if(tensor_max[i] > max_tensor_value)
      max_tensor_value = tensor_max[i];
  }

  step_value = min_double(c->x_step, c->y_step);
  min_step_value = min_double(step_value, c->z_step);

  //dt = 1/(6*max_tensor_value*(min_step_value*min_step_value));
  dt = (min_step_value*min_step_value)/(6*max_tensor_value);
  printf("dt < %f \n", dt);
  return dt;
}

void enforce_grid_left_right(double ***E, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i <= z+2; i++)
  {
    for(j = 0; j <= y+2; j++)
    {
      for(k = 0; k <= x+2; k+=x+2)
      {
        E[i][j][k] = 0.0;
      }
    }
  }
}

void enforce_grid_up_down(double ***E, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i <= z+2; i++)
  {
    for(j = 0; j <= y+2; j+=y+2)
    {
      for(k = 0; k <= x+2; k++)
      {
        E[i][j][k] = 0.0;
      }
    }
  }
}

void enforce_grid_zup_zdown(double ***E, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i <= z+2; i+=z+2)
  {
    for(j = 0; j <= y+2; j++)
    {
      for(k = 0; k <= x+2; k++)
      {
        E[i][j][k] = 0.0;
      }
    }
  }
}
