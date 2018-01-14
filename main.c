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

  /*Read from tensor files*/
  read_binaryformat("all/mesh_new/tensor_val11_16.tensor", &c->tensor_val11, c->nxc+2, c->nyc+2, c->nzc+2);
  read_binaryformat("all/mesh_new/tensor_val12_16.tensor", &c->tensor_val12, c->nxc+2, c->nyc+2, c->nzc+2);
  read_binaryformat("all/mesh_new/tensor_val13_16.tensor", &c->tensor_val13, c->nxc+2, c->nyc+2, c->nzc+2);
  read_binaryformat("all/mesh_new/tensor_val22_16.tensor", &c->tensor_val22, c->nxc+2, c->nyc+2, c->nzc+2);
  read_binaryformat("all/mesh_new/tensor_val23_16.tensor", &c->tensor_val23, c->nxc+2, c->nyc+2, c->nzc+2);
  read_binaryformat("all/mesh_new/tensor_val33_16.tensor", &c->tensor_val33, c->nxc+2, c->nyc+2, c->nzc+2);

  double W = 1.0;
  double pi = 3.14159265358979323846;
  int x_outside = 0;
  int y_inside = 0;
  for(i = 1; i <= c->nzc; i++)
  {
    for(j = 1; j <= c->nyc; j++)
    {
      for(k = 1; k <= c->nxc; k++)
      {
        c->u_old[i][j][k] = exp(0*3*W*W*pi*pi)*cos(pi*c->grid_x[k]*W)*cos(pi*c->grid_y[j]*W)*cos(pi*c->grid_z[i]*W);
        c->u_new[i][j][k] = exp(0*3*W*W*pi*pi)*cos(pi*c->grid_x[k]*W)*cos(pi*c->grid_y[j]*W)*cos(pi*c->grid_z[i]*W);
        count++;
      }
    }
  }

  //printf("Total inside points %d, of: %d \n", y_inside, c->nxc*c->nyc*c->nzc);

  double norm_arr = 0;
  double norm_sum = 0;
  for(i = 1; i <= c->nzc; i++)
  {
    for(j = 1; j <= c->nyc; j++)
    {
      for(k = 1; k <= c->nxc; k++)
      {
        //norm_arr += (c->u_old[i][j][k]*c->u_old[i][j][k]);
        norm_arr += (c->u_old[i][j][k]*c->u_old[i][j][k]);
      }
    }
  }

  printf("Before simulation: %f \n", sqrt(norm_arr));


  //double dt = 0.00001;
  //double dt_test = 0.1;
  //double dt = compute_dt(c);
  double dt = 0.9;//dt_test;
  //double dt_final = dt_test/1000;
  double debug = 0;
  double norm_test = 0;
  double norm_time = 0;
  int time_step = 10;
  double ti = 100;
  int l;
  double gradient_x = 0;
  double gradient_y = 0;
  double gradient_z = 0;
  double divergence_x = 0;
  double divergence_y = 0;
  double divergence_z = 0;
  double alpha = 1/(4*c->x_step);
  double beta = 1/(4*c->y_step);
  double ghamma = 1/(4*c->z_step);
  double rho_x = dt*alpha;
  double rho_y = dt*beta;
  double rho_z = dt*ghamma;
  double ***temp;
  int bx = c->nxc;
  int by = c->nyc/4;
  int bz = c->nzc/8;
  int iz, jy, kx;

  double start1 = omp_get_wtime();
  #ifdef REG
  for(l = 1; l <= time_step; l++)
  {

    for(i = 1; i <= z+1; i++)
    {
      for(j = 1; j <= y+1; j++)
      {
        for(k = 1; k <= x+1; k++)
        {
          /*c->u_new[i][j][k] = c->u_old[i][j][k] + dt*(divergence_cell_direction_x(c->u_old, kx, 0, 0, c->x_step, c->y_step, c->z_step, i, j, k)
                            + divergence_cell_direction_y(c->u_old, 0, kx, 0, c->x_step, c->y_step, c->z_step, i, j, k)
                            + divergence_cell_direction_z(c->u_old, 0, 0, kx, c->x_step, c->y_step, c->z_step, i, j, k));*/

          //c->u_debug[i][j][k] = exp(-dt*3*W*W*pi*pi*kx)*cos(pi*c->grid_x[k]*W)*cos(pi*c->grid_y[j]*W)*cos(pi*c->grid_z[i]*W);

          //norm_test += norm(c->u_new[i][j][k] - c->u_debug[i][j][k]);
          //printf("norm: %f \n", norm_test);
          //norm_test += (c->u_new[i][j][k] - c->u_debug[i][j][k])*(c->u_new[i][j][k] - c->u_debug[i][j][k]);
          //norm_test += (c->u_new[i][j][k]*c->u_new[i][j][k]);

          c->u_new[i][j][k] = c->u_old[i][j][k] + (dt*(diffusion(c->u_old, kx, kx, kx,
            kx, kx, kx,
            kx, kx, kx,
            c->x_step, c->y_step, c->z_step, i, j, k)));

          /*c->u_new[i][j][k] = (dt*(diffusion(c->u_old, c->tensor_val11[i][j][k], c->tensor_val12[i][j][k], c->tensor_val13[i][j][k],
            c->tensor_val12[i][j][k], c->tensor_val22[i][j][k], c->tensor_val23[i][j][k],
            c->tensor_val13[i][j][k], c->tensor_val23[i][j][k], c->tensor_val33[i][j][k],
            c->x_step, c->y_step, c->z_step, i, j, k)));*/

            //c->u_new[i][j][k] = c->u_old[i][j][k] + (dt*(divergence_cell_direction_x(c->u_old, c->tensor_val11[i][j][k], c->tensor_val12[i][j][k], c->tensor_val13[i][j][k], c->x_step, c->y_step, c->y_step, i, j, k)
                              //+ divergence_cell_direction_y(c->u_old, c->tensor_val12[i][j][k], c->tensor_val22[i][j][k], c->tensor_val23[i][j][k], c->x_step, c->y_step, c->y_step, i, j, k)
                              //+ divergence_cell_direction_z(c->u_old, c->tensor_val13[i][j][k], c->tensor_val23[i][j][k], c->tensor_val33[i][j][k], c->x_step, c->y_step, c->y_step, i, j, k)));
        }
     }
   }

   /*enforce_grid_up_down(c->u_new, x, y, z);
   enforce_grid_left_right(c->u_new, x, y, z);
   enforce_grid_zup_zdown(c->u_new, x, y, z);*/
   //printf("%f \n", norm_test);

   /*for(i = 1; i <= z+1; i++)
   {
     for(j = 1; j <= y+1; j++)
     {
       for(k = 1; k <= x+1; k++)
       {
         norm_test += (c->u_new[i][j][k]*c->u_new[i][j][k]);
       }
     }
   }

   //printf("norm_test: %0.12f \n", norm_test);
   norm_time += norm_test;*/

   ti = l;
   //norm_test += c->u_diff[i][j][k];

   double ***temp;
   temp = c->u_old;
   c->u_old = c->u_new;
   c->u_new = temp;
  }
  #endif

  #ifdef NEW
  for(l = 1; l <= time_step; l++)
  {
    for(i = 1; i <= c->nzc; i++)
    {
      for(j = 1; j <= c->nyc; j++)
      {
        for(k = 1; k <= c->nxc; k++)
        {
          upper_right_one_x = alpha*(c->u_old[i+1][j+1][k+1] + c->u_old[i+1][j][k+1] + c->u_old[i+1][j+1][k] + c->u_old[i+1][j][k] -
                        c->u_old[i][j+1][k+1] - c->u_old[i][j][k+1] - c->u_old[i][j+1][k] - c->u_old[i][j][k]);
          upper_right_one_y = beta*(c->u_old[i+1][j+1][k+1] + c->u_old[i][j+1][k+1] + c->u_old[i+1][j+1][k] + c->u_old[i][j+1][k] -
                        c->u_old[i+1][j][k+1] - c->u_old[i][j][k+1] - c->u_old[i+1][j][k] - c->u_old[i][j][k]);
          upper_right_one_z = ghamma*(c->u_old[i+1][j+1][k+1] + c->u_old[i+1][j][k+1] + c->u_old[i][j+1][k+1] + c->u_old[i][j][k+1] -
                        c->u_old[i+1][j+1][k] - c->u_old[i+1][j][k] - c->u_old[i][j+1][k] - c->u_old[i][j][k]);

          lower_right_one_x = alpha*(c->u_old[i+1][j][k+1] + c->u_old[i+1][j-1][k+1] + c->u_old[i+1][j][k] + c->u_old[i+1][j-1][k] -
                        c->u_old[i][j][k+1] - c->u_old[i][j-1][k+1] - c->u_old[i][j][k] - c->u_old[i][j-1][k]);
          lower_right_one_y = beta*(c->u_old[i+1][j][k+1] + c->u_old[i][j][k+1] + c->u_old[i+1][j][k] + c->u_old[i][j][k] -
                        c->u_old[i+1][j-1][k+1] - c->u_old[i][j-1][k+1] - c->u_old[i+1][j-1][k] - c->u_old[i][j-1][k]);
          lower_right_one_z = ghamma*(c->u_old[i+1][j][k+1] + c->u_old[i+1][j-1][k+1] + c->u_old[i][j][k+1] + c->u_old[i][j-1][k+1] -
                        c->u_old[i+1][j][k] - c->u_old[i+1][j-1][k] - c->u_old[i][j][k] - c->u_old[i][j-1][k]);

          upper_right_x = alpha*(c->u_old[i+1][j+1][k] + c->u_old[i+1][j][k] + c->u_old[i+1][j+1][k-1] + c->u_old[i+1][j][k-1] -
                        c->u_old[i][j+1][k] - c->u_old[i][j][k] - c->u_old[i][j+1][k-1] - c->u_old[i][j][k-1]);
          upper_right_y = beta*(c->u_old[i+1][j+1][k] + c->u_old[i][j+1][k] + c->u_old[i+1][j+1][k-1] + c->u_old[i][j+1][k-1] -
                        c->u_old[i+1][j][k] - c->u_old[i][j][k] - c->u_old[i+1][j][k-1] - c->u_old[i][j][k-1]);
          upper_right_z = ghamma*(c->u_old[i+1][j+1][k] + c->u_old[i+1][j][k] + c->u_old[i][j+1][k] + c->u_old[i][j][k] -
                        c->u_old[i+1][j+1][k-1] - c->u_old[i+1][j][k-1] - c->u_old[i][j+1][k-1] - c->u_old[i][j][k-1]);

          lower_right_x = alpha*(c->u_old[i+1][j][k] + c->u_old[i+1][j-1][k] + c->u_old[i+1][j][k-1] + c->u_old[i+1][j-1][k-1] -
                        c->u_old[i][j][k] - c->u_old[i][j-1][k] - c->u_old[i][j][k-1] - c->u_old[i][j-1][k-1]);
          lower_right_y = beta*(c->u_old[i+1][j][k] + c->u_old[i][j][k] + c->u_old[i+1][j][k-1] + c->u_old[i][j][k-1] -
                        c->u_old[i+1][j-1][k] - c->u_old[i][j-1][k] - c->u_old[i+1][j-1][k-1] - c->u_old[i][j-1][k-1]);
          lower_right_z = ghamma*(c->u_old[i+1][j][k] + c->u_old[i+1][j-1][k] + c->u_old[i][j][k] + c->u_old[i][j-1][k] -
                        c->u_old[i+1][j][k-1] - c->u_old[i+1][j-1][k-1] - c->u_old[i][j][k-1] - c->u_old[i][j-1][k-1]);

          upper_left_one_x = alpha*(c->u_old[i][j+1][k+1] + c->u_old[i][j][k+1] + c->u_old[i][j+1][k] + c->u_old[i][j][k] -
                        c->u_old[i-1][j+1][k+1] - c->u_old[i-1][j][k+1] - c->u_old[i-1][j+1][k] - c->u_old[i-1][j][k]);
          upper_left_one_y = beta*(c->u_old[i][j+1][k+1] + c->u_old[i-1][j+1][k+1] + c->u_old[i][j+1][k] + c->u_old[i-1][j+1][k] -
                        c->u_old[i][j][k+1] - c->u_old[i-1][j][k+1] - c->u_old[i][j][k] - c->u_old[i-1][j][k]);
          upper_left_one_z = ghamma*(c->u_old[i][j+1][k+1] + c->u_old[i][j][k+1] + c->u_old[i-1][j+1][k+1] + c->u_old[i-1][j][k+1] -
                        c->u_old[i][j+1][k] - c->u_old[i][j][k] - c->u_old[i-1][j+1][k] - c->u_old[i-1][j][k]);

          lower_left_one_x = alpha*(c->u_old[i][j][k+1] + c->u_old[i][j-1][k+1] + c->u_old[i][j][k] + c->u_old[i][j-1][k] -
                        c->u_old[i-1][j][k+1] - c->u_old[i-1][j-1][k+1] - c->u_old[i-1][j][k] - c->u_old[i-1][j-1][k]);
          lower_left_one_y = beta*(c->u_old[i][j][k+1] + c->u_old[i-1][j][k+1] + c->u_old[i][j][k] + c->u_old[i-1][j][k] -
                        c->u_old[i][j-1][k+1] - c->u_old[i-1][j-1][k+1] - c->u_old[i][j-1][k] - c->u_old[i-1][j-1][k]);
          lower_left_one_z = ghamma*(c->u_old[i][j][k+1] + c->u_old[i][j-1][k+1] + c->u_old[i-1][j][k+1] + c->u_old[i-1][j-1][k+1] -
                        c->u_old[i][j][k] - c->u_old[i][j-1][k] - c->u_old[i-1][j][k] - c->u_old[i-1][j-1][k]);

          upper_left_x = alpha*(c->u_old[i][j+1][k] + c->u_old[i][j][k] + c->u_old[i][j+1][k-1] + c->u_old[i][j][k-1] -
                        c->u_old[i-1][j+1][k] - c->u_old[i-1][j][k] - c->u_old[i-1][j+1][k-1] - c->u_old[i-1][j][k-1]);
          upper_left_y = beta*(c->u_old[i][j+1][k] + c->u_old[i-1][j+1][k] + c->u_old[i][j+1][k-1] + c->u_old[i-1][j+1][k-1] -
                        c->u_old[i][j][k] - c->u_old[i-1][j][k] - c->u_old[i][j][k-1] - c->u_old[i-1][j][k-1]);
          upper_left_z = ghamma*(c->u_old[i][j+1][k] + c->u_old[i][j][k] + c->u_old[i-1][j+1][k] + c->u_old[i-1][j][k] -
                        c->u_old[i][j+1][k-1] - c->u_old[i][j][k-1] - c->u_old[i-1][j+1][k-1] - c->u_old[i-1][j][k-1]);

          lower_left_x = alpha*(c->u_old[i][j][k] + c->u_old[i][j-1][k] + c->u_old[i][j][k-1] + c->u_old[i][j-1][k-1] -
                        c->u_old[i-1][j][k] - c->u_old[i-1][j-1][k] - c->u_old[i-1][j][k-1] - c->u_old[i-1][j-1][k-1]);
          lower_left_y = beta*(c->u_old[i][j][k] + c->u_old[i-1][j][k] + c->u_old[i][j][k-1] + c->u_old[i-1][j][k-1] -
                        c->u_old[i][j-1][k] - c->u_old[i-1][j-1][k] - c->u_old[i][j-1][k-1] - c->u_old[i-1][j-1][k-1]);
          lower_left_z = ghamma*(c->u_old[i][j][k] + c->u_old[i][j-1][k] + c->u_old[i-1][j][k] + c->u_old[i-1][j-1][k] -
                        c->u_old[i][j][k-1] - c->u_old[i][j-1][k-1] - c->u_old[i-1][j][k-1] - c->u_old[i-1][j-1][k-1]);

          diffusion_x =
          ((upper_right_one_x*c->tensor_val11[i][j][k] + upper_right_one_y*c->tensor_val12[i][j][k] + upper_right_one_z*c->tensor_val13[i][j][k])
          + (lower_right_one_x*c->tensor_val11[i][j-1][k] + lower_right_one_y*c->tensor_val12[i][j-1][k] + lower_right_one_z*c->tensor_val13[i][j-1][k])
          + (upper_right_x*c->tensor_val11[i][j][k-1] + upper_right_y*c->tensor_val12[i][j][k-1] + upper_right_z*c->tensor_val13[i][j][k-1])
          + (lower_right_x*c->tensor_val11[i][j-1][k-1] + lower_right_y*c->tensor_val12[i][j-1][k-1] + lower_right_z*c->tensor_val13[i][j-1][k-1])
          - (upper_left_one_x*c->tensor_val11[i-1][j][k] + upper_left_one_y*c->tensor_val12[i-1][j][k] + upper_left_one_z*c->tensor_val13[i-1][j][k])
          - (lower_left_one_x*c->tensor_val11[i-1][j-1][k] + lower_left_one_y*c->tensor_val12[i-1][j-1][k] + lower_left_one_z*c->tensor_val13[i-1][j-1][k])
          - (upper_left_x*c->tensor_val11[i-1][j][k-1] + upper_left_y*c->tensor_val12[i-1][j][k-1] + upper_left_z*c->tensor_val13[i-1][j][k-1])
          - (lower_left_x*c->tensor_val11[i-1][j-1][k-1] + lower_left_y*c->tensor_val12[i-1][j-1][k-1] + lower_left_z*c->tensor_val13[i-1][j-1][k-1]));

          diffusion_y =
          ((upper_right_one_x*c->tensor_val12[i][j][k] + upper_right_one_y*c->tensor_val22[i][j][k] + upper_right_one_z*c->tensor_val23[i][j][k])
          + (upper_left_one_x*c->tensor_val12[i-1][j][k] + upper_left_one_y*c->tensor_val12[i-1][j][k] + upper_left_one_z*c->tensor_val12[i-1][j][k])
          + (upper_right_x*c->tensor_val12[i][j][k-1] + upper_right_y*c->tensor_val12[i][j][k-1] + upper_right_z*c->tensor_val12[i][j][k-1])
          + (upper_left_x*c->tensor_val12[i-1][j][k-1] + upper_left_y*c->tensor_val12[i-1][j][k-1] + upper_left_z*c->tensor_val12[i-1][j][k-1])
          - (lower_right_one_x*c->tensor_val12[i][j-1][k] + lower_right_one_y*c->tensor_val12[i][j-1][k] + lower_right_one_z*c->tensor_val12[i][j-1][k])
          - (lower_left_one_x*c->tensor_val12[i-1][j-1][k] + lower_left_one_y*c->tensor_val12[i-1][j-1][k] + lower_left_one_z*c->tensor_val12[i-1][j-1][k])
          - (lower_right_x*c->tensor_val12[i][j-1][k-1] + lower_right_y*c->tensor_val12[i][j-1][k-1] + lower_right_z*c->tensor_val12[i][j-1][k-1])
          - (lower_left_x*c->tensor_val12[i-1][j-1][k-1] + lower_left_y*c->tensor_val12[i-1][j-1][k-1] + lower_left_z*c->tensor_val12[i-1][j-1][k-1]));

          diffusion_z =
          ((upper_right_one_x*c->tensor_val13[i][j][k] + upper_right_one_y*c->tensor_val23[i][j][k] + upper_right_one_z*c->tensor_val33[i][j][k])
          + (lower_right_one_x*c->tensor_val13[i][j-1][k] + lower_right_one_y*c->tensor_val13[i][j-1][k] + lower_right_one_z*c->tensor_val13[i][j-1][k])
          + (upper_left_one_x*c->tensor_val13[i-1][j][k] + upper_left_one_y*c->tensor_val13[i-1][j][k] + upper_left_one_z*c->tensor_val13[i-1][j][k])
          + (lower_left_one_x*c->tensor_val13[i-1][j-1][k] + lower_left_one_y*c->tensor_val13[i-1][j-1][k] + lower_left_one_z*c->tensor_val13[i-1][j-1][k])
          - (upper_right_x*c->tensor_val13[i][j][k-1] + upper_right_y*c->tensor_val13[i][j][k-1] + upper_right_z*c->tensor_val13[i][j][k-1])
          - (lower_right_x*c->tensor_val13[i][j-1][k-1] + lower_right_y*c->tensor_val13[i][j-1][k-1] + lower_right_z*c->tensor_val13[i][j-1][k-1])
          - (upper_left_x*c->tensor_val13[i-1][j][k-1] + upper_left_y*c->tensor_val13[i-1][j][k-1] + upper_left_z*c->tensor_val13[i-1][j][k-1])
          - (lower_left_x*c->tensor_val13[i-1][j-1][k-1] + lower_left_y*c->tensor_val13[i-1][j-1][k-1] + lower_left_z*c->tensor_val13[i-1][j-1][k-1]));

          c->u_new[i][j][k] = c->u_old[i][j][k] + (rho_x*diffusion_x + rho_y*diffusion_y + rho_z*diffusion_z);
        }
      }
    }

   temp = c->u_old;
   c->u_old = c->u_new;
   c->u_new = temp;
  }
  #endif

  #ifdef TEST
  for(l = 0; l < time_step; l++)
  {
    for(i = 1; i <= c->nzc; i++)
    {
      for(j = 1; j <= c->nyc; j++)
      {
        for(k = 1; k <= c->nxc; k++)
        {
          lower_left = c->u_old[i][j][k];
          upper_right_one = c->u_old[i+1][j+1][k+1];
          lower_right_one = c->u_old[i+1][j][k+1];
          upper_right = c->u_old[i+1][j+1][k];
          lower_right = c->u_old[i+1][j][k];
          upper_left_one = c->u_old[i][j+1][k+1];
          lower_left_one = c->u_old[i][j][k+1];
          upper_left = c->u_old[i][j+1][k];

          gradient_x = alpha*(upper_right_one + lower_right_one + upper_right + lower_right -
                        upper_left_one - lower_left_one - upper_left - lower_left);

          gradient_y = beta*(upper_right_one + upper_left_one + upper_right + upper_left -
                        lower_right_one - lower_left_one - lower_right - lower_left);

          gradient_z = ghamma*(upper_right_one + lower_right_one + upper_left_one + lower_left_one -
                        upper_right - lower_right - upper_left - lower_left);

          c->flux_x[i][j][k] = (c->tensor_val11[i][j][k]*gradient_x) + (c->tensor_val12[i][j][k]*gradient_y)
          + (c->tensor_val13[i][j][k]*gradient_z);

          c->flux_y[i][j][k] = (c->tensor_val12[i][j][k]*gradient_x) + (c->tensor_val22[i][j][k]*gradient_y)
          + (c->tensor_val23[i][j][k]*gradient_z);

          c->flux_z[i][j][k] = (c->tensor_val13[i][j][k]*gradient_x) + (c->tensor_val23[i][j][k]*gradient_y)
          + (c->tensor_val33[i][j][k]*gradient_z);

        }
      }
    }

    for(i = 1; i <= c->nzc; i++)
    {
      for(j = 1; j <= c->nyc; j++)
      {
        for(k = 1; k <= c->nxc; k++)
        {
          diffusion_x = (c->flux_x[i][j][k] + c->flux_x[i][j-1][k] + c->flux_x[i][j][k-1] + c->flux_x[i][j-1][k-1] -
                        c->flux_x[i-1][j][k] - c->flux_x[i-1][j-1][k] - c->flux_x[i-1][j][k-1] - c->flux_x[i-1][j-1][k-1]);

          diffusion_y = (c->flux_y[i][j][k] + c->flux_y[i-1][j][k] + c->flux_y[i][j][k-1] + c->flux_y[i-1][j][k-1] -
                        c->flux_y[i][j-1][k] - c->flux_y[i-1][j-1][k] - c->flux_y[i][j-1][k-1] - c->flux_y[i-1][j-1][k-1]);

          diffusion_z = (c->flux_z[i][j][k] + c->flux_z[i][j-1][k] + c->flux_z[i-1][j][k] + c->flux_z[i-1][j-1][k] -
                        c->flux_z[i][j][k-1] - c->flux_z[i][j-1][k-1] - c->flux_z[i-1][j][k-1] - c->flux_z[i-1][j-1][k-1]);

          c->u_new[i][j][k] = c->u_old[i][j][k] + (rho_x*diffusion_x + rho_y*diffusion_y + rho_z*diffusion_z);
        }
      }
    }

   temp = c->u_old;
   c->u_old = c->u_new;
   c->u_new = temp;
  }
  #endif

  #ifdef BLOCK

  for(l = 0; l < time_step; l++)
  {
    //min(iz+bz, c->nzc)
    for(iz = 1; iz < c->nzc; iz+=bz)
    {
      for(jy = 1; jy < c->nyc; jy+=by)
      {
        for(kx = 1; kx < c->nxc; kx+=bx)
        {
          //printf("iz: %d, iz+(bz-1): %d, (iz+(bz-1))-1: %d", iz, iz+(bz), (iz+(bz))-1);
          for(i = iz; i <= min(iz+(bz), c->nzc); i++)
          {
            for(j = jy; j <= min(jy+(by), c->nyc); j++)
            {
              for(k = kx; k <= min(kx+(bx), c->nxc); k++)
              {
                lower_left = c->u_old[i][j][k];
                upper_right_one = c->u_old[i+1][j+1][k+1];
                lower_right_one = c->u_old[i+1][j][k+1];
                upper_right = c->u_old[i+1][j+1][k];
                lower_right = c->u_old[i+1][j][k];
                upper_left_one = c->u_old[i][j+1][k+1];
                lower_left_one = c->u_old[i][j][k+1];
                upper_left = c->u_old[i][j+1][k];

                gradient_x = alpha*(upper_right_one + lower_right_one + upper_right + lower_right -
                              upper_left_one - lower_left_one - upper_left - lower_left);

                gradient_y = beta*(upper_right_one + upper_left_one + upper_right + upper_left -
                              lower_right_one - lower_left_one - lower_right - lower_left);

                gradient_z = ghamma*(upper_right_one + lower_right_one + upper_left_one + lower_left_one -
                              upper_right - lower_right - upper_left - lower_left);


                c->flux_x[i][j][k] = (c->tensor_val11[i][j][k]*gradient_x) + (c->tensor_val12[i][j][k]*gradient_y)
                + (c->tensor_val13[i][j][k]*gradient_z);

                c->flux_y[i][j][k] = (c->tensor_val12[i][j][k]*gradient_x) + (c->tensor_val22[i][j][k]*gradient_y)
                + (c->tensor_val23[i][j][k]*gradient_z);

                c->flux_z[i][j][k] = (c->tensor_val13[i][j][k]*gradient_x) + (c->tensor_val23[i][j][k]*gradient_y)
                + (c->tensor_val33[i][j][k]*gradient_z);

                diffusion_x = (c->flux_x[i][j][k] + c->flux_x[i][j-1][k] + c->flux_x[i][j][k-1] + c->flux_x[i][j-1][k-1] -
                              c->flux_x[i-1][j][k] - c->flux_x[i-1][j-1][k] - c->flux_x[i-1][j][k-1] - c->flux_x[i-1][j-1][k-1]);

                diffusion_y = (c->flux_y[i][j][k] + c->flux_y[i-1][j][k] + c->flux_y[i][j][k-1] + c->flux_y[i-1][j][k-1] -
                              c->flux_y[i][j-1][k] - c->flux_y[i-1][j-1][k] - c->flux_y[i][j-1][k-1] - c->flux_y[i-1][j-1][k-1]);

                diffusion_z = (c->flux_z[i][j][k] + c->flux_z[i][j-1][k] + c->flux_z[i-1][j][k] + c->flux_z[i-1][j-1][k] -
                              c->flux_z[i][j][k-1] - c->flux_z[i][j-1][k-1] - c->flux_z[i-1][j][k-1] - c->flux_z[i-1][j-1][k-1]);

                c->u_new[i][j][k] = c->u_old[i][j][k] + (rho_x*diffusion_x + rho_y*diffusion_y + rho_z*diffusion_z);
              }
            }
          }

           /*for(i = iz; i <= min(iz+(bz), c->nzc); i++)
           {
             for(j = jy; j <= min(jy+(by), c->nyc); j++)
             {
               for(k = kx; k <= min(kx+(bx), c->nxc); k++)
               {
                 diffusion_x = (c->flux_x[i][j][k] + c->flux_x[i][j-1][k] + c->flux_x[i][j][k-1] + c->flux_x[i][j-1][k-1] -
                               c->flux_x[i-1][j][k] - c->flux_x[i-1][j-1][k] - c->flux_x[i-1][j][k-1] - c->flux_x[i-1][j-1][k-1]);

                 diffusion_y = (c->flux_y[i][j][k] + c->flux_y[i-1][j][k] + c->flux_y[i][j][k-1] + c->flux_y[i-1][j][k-1] -
                               c->flux_y[i][j-1][k] - c->flux_y[i-1][j-1][k] - c->flux_y[i][j-1][k-1] - c->flux_y[i-1][j-1][k-1]);

                 diffusion_z = (c->flux_z[i][j][k] + c->flux_z[i][j-1][k] + c->flux_z[i-1][j][k] + c->flux_z[i-1][j-1][k] -
                               c->flux_z[i][j][k-1] - c->flux_z[i][j-1][k-1] - c->flux_z[i-1][j][k-1] - c->flux_z[i-1][j-1][k-1]);

                 c->u_new[i][j][k] = c->u_old[i][j][k] + (rho_x*diffusion_x + rho_y*diffusion_y + rho_z*diffusion_z);
               }
            }
          }*/
        }
       }
     }

     /*for(i = 1; i <= c->nzc; i++)
     {
       for(j = 1; j <= c->nyc; j++)
       {
         for(k = 1; k <= c->nxc; k++)
         {
           diffusion_x = (c->flux_x[i][j][k] + c->flux_x[i][j-1][k] + c->flux_x[i][j][k-1] + c->flux_x[i][j-1][k-1] -
                         c->flux_x[i-1][j][k] - c->flux_x[i-1][j-1][k] - c->flux_x[i-1][j][k-1] - c->flux_x[i-1][j-1][k-1]);

           diffusion_y = (c->flux_y[i][j][k] + c->flux_y[i-1][j][k] + c->flux_y[i][j][k-1] + c->flux_y[i-1][j][k-1] -
                         c->flux_y[i][j-1][k] - c->flux_y[i-1][j-1][k] - c->flux_y[i][j-1][k-1] - c->flux_y[i-1][j-1][k-1]);

           diffusion_z = (c->flux_z[i][j][k] + c->flux_z[i][j-1][k] + c->flux_z[i-1][j][k] + c->flux_z[i-1][j-1][k] -
                         c->flux_z[i][j][k-1] - c->flux_z[i][j-1][k-1] - c->flux_z[i-1][j][k-1] - c->flux_z[i-1][j-1][k-1]);

           c->u_new[i][j][k] = c->u_old[i][j][k] + (rho_x*diffusion_x + rho_y*diffusion_y + rho_z*diffusion_z);
           //c->u_new[i][j][k] = dt*(diffusion_x);//(2*c->x_step);
           //c->u_new[i][j][k] = c->u_old[i][j][k] + c->flux_x[i][j][k];
         }
       }
     }*/

   temp = c->u_old;
   c->u_old = c->u_new;
   c->u_new = temp;
  }
  #endif

  double end = omp_get_wtime() - start1;
  double norm_debug = 0.0;
  for(i = 1; i <= c->nzc; i++)
  {
    for(j = 1; j <= c->nyc; j++)
    {
      for(k = 1; k <= c->nxc; k++)
      {
        norm_test += (c->u_new[i][j][k]*c->u_new[i][j][k]);
      }
    }
  }
  norm_time += norm_test;

  int is_inside = 0;
  int is_outside = 0;
  #ifdef BLOCK
  printf("Blocked version of size: %d x %d x %d \n", bx, by, bz);
  #endif
  //printf("inside: %d  outside: %d\n", is_inside, is_outside);
  printf("Size: %dx%dx%d\n", c->nxc, c->nyc, c->nzc);
  printf("Time taken: %fs\n", end);
  printf("Time step: %d\n", time_step);
  printf("GFLOPS: %f \n", 66*time_step*(double)((x)*(y)*(z))*1e-9/end);
  printf("Memory bandwidth v8: %f GB/s\n", (8*time_step*(double)(c->nxc*c->nyc*c->nzc)*8*1e-9)/end);
  printf("Memory bandwidth v9: %f GB/s\n", (9*time_step*(double)(c->nxc*c->nyc*c->nzc)*8*1e-9)/end);
  printf("Memory bandwidth v10: %f GB/s\n", (10*time_step*(double)(c->nxc*c->nyc*c->nzc)*8*1e-9)/end);
  printf("Memory bandwidth v11: %f GB/s\n", (11*time_step*(double)(c->nxc*c->nyc*c->nzc)*8*1e-9)/end);
  printf("Memory bandwidth v14: %f GB/s\n", (14*time_step*(double)(c->nxc*c->nyc*c->nzc)*8*1e-9)/end);
  printf("Memory bandwidth v15: %f GB/s\n", (15*time_step*(double)(c->nxc*c->nyc*c->nzc)*8*1e-9)/end);
  printf("Points per second: %f *10ˆ9\n", (((double)(c->nxc*c->nyc*c->nzc))*1e-9*time_step)/end);
  printf("Cells per second in bytes: %f *10ˆ9\n", ((double)(c->nxc*c->nyc*c->nzc)*1e-9*time_step*8*8)/end);
  printf("Cells per second: %f *10ˆ9\n", ((double)(c->nxc*c->nyc*c->nzc)*1e-9*time_step*8)/end);
  printf("Result numerical: %f \n", sqrt(norm_test));

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

  c->u_debug = dallocate_3d(x+2, y+2, z+2);
  dinit_3d(c->u_debug, x+2, y+2, z+2);

  c->u_diff = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->u_diff, x+3, y+3, z+3);

  c->tensor_dummy = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->tensor_dummy, x+3, y+3, z+3);

  c->gradient_x = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->gradient_x, x+3, y+3, z+3);

  c->gradient_y = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->gradient_y, x+3, y+3, z+3);

  c->gradient_z = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->gradient_z, x+3, y+3, z+3);

  /*c->c->tensor_val11[i][j][k]0 = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->c->tensor_val11[i][j][k]0, x+3, y+3, z+3);

  c->c->tensor_val11[i][j][k]1 = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->c->tensor_val11[i][j][k]1, x+3, y+3, z+3);

  c->c->tensor_val12[i][j][k]0 = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->c->tensor_val12[i][j][k]0, x+3, y+3, z+3);

  c->c->tensor_val12[i][j][k]1 = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->c->tensor_val12[i][j][k]1, x+3, y+3, z+3);

  c->c->tensor_val13[i][j][k]0 = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->c->tensor_val13[i][j][k]0, x+3, y+3, z+3);

  c->c->tensor_val13[i][j][k]1 = dallocate_3d(x+3, y+3, z+3);
  dinit_3d(c->c->tensor_val13[i][j][k]1, x+3, y+3, z+3);*/

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
