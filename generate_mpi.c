#include <stdio.h>
#include "tensor.h"
#include "mesh.h"
#include "io.h"
#include "diffusion.h"
//#include "omp.h"
#include "math.h"
#include "mpi.h"
#include "omp.h"

#define TAG1 111
#define TAG2 222
#define TAG3 333
#define TAG4 444
#define TAG5 555
#define TAG6 666

double ***dallocate_3d(int x, int y, int z);
void dinit_3d(double*** matrix, int x, int y, int z);
void init_mpi_data(cube *c, int x, int y, int z, int nx, int ny, int nz,
  int x0, int x1, int y0, int y1, int z0, int z1);
void enforce_grid_left_right(double ***E, int x, int y, int z);
void enforce_grid_up_down(double ***E, int x, int y, int z);
void enforce_grid_zup_zdown(double ***E, int x, int y, int z);
void decompose(int n, int dim, int coord, int* start, int* end);
void free3d_global(double ***arr, cube *c);
void free3d_local(double ***arr, cube *c);
void free1d(double *arr);
void free_data(cube *c);
void initial_condition(cube *c, int start_x, int start_y, int start_z, int end_x, int end_y, int end_z);

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
  int i, j, k;
  int iz, jy, kxx;
  int ii, jj, kk;
  int nx, ny, nz;
  int start_x, start_y, start_z;
  int end_x, end_y, end_z;
  int size, rank;
  int coords[3];
  int periods[3];
  int dims[3];
  int dimx[3];
  int procs_x, procs_y, procs_z;
  int left, right, up, down, z_up, z_down;
  int x0, x1, y0, y1, z0, z1;
  double count = 1.0;
  int count_inside, count_outside;
  int count_inside_tensor, count_outside_tensor;
  double l2_diff = 0;
  double l2_norm = 0;
  double kx;
  double gradient_x, gradient_y, gradient_z;
  double diffusion_x, diffusion_y, diffusion_z;
  //double upper_right_one, lower_right_one, upper_right, lower_right, upper_left_one, lower_left_one, upper_left, lower_left;
  double lower_left = 0.0;
  double upper_right_one = 0.0;
  double lower_right_one = 0.0;
  double upper_right = 0.0;
  double lower_right = 0.0;
  double upper_left_one = 0.0;
  double lower_left_one = 0.0;
  double upper_left = 0.0;
  double a[3], b[3], cc[3], d[3];
  cube *c = (cube*)malloc(sizeof(cube));
  meshdata *m = (meshdata*)malloc(sizeof(meshdata));
  tensorfield *t = (tensorfield *)malloc(sizeof(tensorfield));

  MPI_Comm comm3d, comm3d_old;
  MPI_Request req[12];
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0)
    printf("mpi ranks: %d \n", size);

  periods[0] = 0;
  periods[1] = 0;
  periods[2] = 0;

  dims[0] = 0;
  dims[1] = 0;
  dims[2] = 0;


  MPI_Dims_create(size, 3, dims);
  if(rank == 0)
    printf("Proc dims: %dx%dx%d \n", dims[0], dims[1], dims[2]);

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm3d);
  MPI_Cart_get(comm3d, 3, dims, periods, coords);
  MPI_Cart_shift(comm3d, 0, 1, &left, &right);
  MPI_Cart_shift(comm3d, 1, 1, &up, &down);
  MPI_Cart_shift(comm3d, 2, 1, &z_up, &z_down);

  decompose((x-1), dims[0], coords[0], &x0, &x1);
  decompose((y-1), dims[1], coords[1], &y0, &y1);
  decompose((z-1), dims[2], coords[2], &z0, &z1);

  start_x = 1;
  start_y = 1;
  start_z = 1;
  end_x = 0;
  end_y = 0;
  end_z = 0;

  nx = x1 - x0 +1;
  ny = y1 - y0 +1;
  nz = z1 - z0 +1;
  //x1 = x1+1;
  if(left >= 0)
  {
    x0--;
    //nx = nx+1;
    start_x = 0;
  }

  if(right >= 0)
  {
    x1++;
    end_x = 1;
  }

  if(up >= 0)
  {
    y0--;
    //ny = ny+1;
    start_y = 0;
  }

  if(down >= 0)
  {
    y1++;
    end_y = 1;
  }

  if(z_up >= 0)
  {
    z0--;
    //nz = nz+1;
    start_z = 0;
  }

  if(z_down >= 0)
  {
    z1++;
    end_z = 1;
  }

  //printf("x0: %d, x1: %d rank: %d start_x: %d nx: %d\n", x0, x1, rank, start_x, nx);

  init_mpi_data(c, x, y, z, nx, ny, nz, x0, x1, y0, y1, z0, z1);
  init_cube_grid_mpi(c, m, start_x, start_y, start_z, rank);

  double start = MPI_Wtime();
  sparse_readtensorfiles("all/mesh_new/3Dheart.1", t, 1000);
  //sparse_readtensorfiles("mesh_new/3Dheart.1", t, 1000);
  fiberstotensors(t);
  generate_tensor_mpi(c, t, m, start_x, start_y, start_z, end_x, end_y, end_z);
  //printf("GENERATE_TENSORS_ONLY\n");
  double end = MPI_Wtime() - start;
  double mpitime = 0;
  MPI_Allreduce(&end,&mpitime,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

  if(rank == 0)
    printf("it took : %0.12f \n", mpitime);

  char buf11[256];
  char buf12[256];
  char buf13[256];
  char buf22[256];
  char buf23[256];
  char buf33[256];
  char str_rank[12];
  char str_mesh_size[12];
  char str_size[12];
  sprintf(str_rank, "%d", rank);
  sprintf(str_size, "%d", size);
  sprintf(str_mesh_size, "%dx%dx%d", c->nxc, c->nyc, c->nzc);
  char *str_name11 = "mpi_mesh/local_tensor_val11";
  char *str_name12 = "mpi_mesh/local_tensor_val12";
  char *str_name13 = "mpi_mesh/local_tensor_val13";
  char *str_name22 = "mpi_mesh/local_tensor_val22";
  char *str_name23 = "mpi_mesh/local_tensor_val23";
  char *str_name33 = "mpi_mesh/local_tensor_val33";
  char *str_end = ".tensor";
  snprintf(buf11, sizeof buf11, "%s_%s_%s_%s%s", str_name11, str_mesh_size, str_rank, str_size, str_end);
  snprintf(buf12, sizeof buf12, "%s_%s_%s_%s%s", str_name12, str_mesh_size, str_rank, str_size, str_end);
  snprintf(buf13, sizeof buf13, "%s_%s_%s_%s%s", str_name13, str_mesh_size, str_rank, str_size, str_end);
  snprintf(buf22, sizeof buf22, "%s_%s_%s_%s%s", str_name22, str_mesh_size, str_rank, str_size, str_end);
  snprintf(buf23, sizeof buf23, "%s_%s_%s_%s%s", str_name23, str_mesh_size, str_rank, str_size, str_end);
  snprintf(buf33, sizeof buf33, "%s_%s_%s_%s%s", str_name33, str_mesh_size, str_rank, str_size, str_end);

  write_binaryformat(buf11, c->local_tensor_val11, nx+2, ny+2, nz+2);
  write_binaryformat(buf12, c->local_tensor_val12, nx+2, ny+2, nz+2);
  write_binaryformat(buf13, c->local_tensor_val13, nx+2, ny+2, nz+2);
  write_binaryformat(buf22, c->local_tensor_val22, nx+2, ny+2, nz+2);
  write_binaryformat(buf23, c->local_tensor_val23, nx+2, ny+2, nz+2);
  write_binaryformat(buf33, c->local_tensor_val33, nx+2, ny+2, nz+2);

  double norm_arr = 0;
  double norm_sum = 0;

  for(i = 1; i <= nz; i++)
  {
    for(j = 1; j <= ny; j++)
    {
      for(k = 1; k <= nx; k++)
      {
        norm_arr += (c->local_tensor_val11[i][j][k]*c->local_tensor_val11[i][j][k]);
      }
    }
  }

  MPI_Allreduce(&norm_arr, &norm_sum, 1, MPI_DOUBLE, MPI_SUM, comm3d);

  MPI_Finalize();

  return 0;
}

void decompose(int n, int dim, int coord, int* start, int* end)
{
  int length, rest;

  length = n/dim;
  rest = n%dim;
  *start = coord * length + (coord < rest ? coord : rest);
  *end = *start + length - (coord < rest ? 0 : 1);
  *start = *start+1;
  *end = *end+1;
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

void init_mpi_data(cube *c, int x, int y, int z, int nx, int ny, int nz, int x0, int x1, int y0, int y1, int z0, int z1)
{
  c->x = x;
  c->y = y;
  c->z = z;
  c->nxc = x-1;
  c->nyc = y-1;
  c->nzc = z-1;
  c->nx = nx;
  c->ny = ny;
  c->nz = nz;
  c->x0 = x0;
  c->x1 = x1;
  c->y0 = y0;
  c->y1 = y1;
  c->z0 = z0;
  c->z1 = z1;
  c->nxc = x-1;
  c->nyc = y-1;
  c->nzc = z-1;

  c->u_old = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->u_old, nx+2, ny+2, nz+2);

  c->u_new = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->u_new, nx+2, ny+2, nz+2);

  /*c->local_tensor = dallocate_3d((nx+2)*6, (ny+2)*6, (nz+2)*6);
  dinit_3d(c->local_tensor, (nx+2)*6, (ny+2)*6, (nz+2)*6);*/

  c->local_tensor_val11 = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->local_tensor_val11, nx+2, ny+2, nz+2);

  c->local_tensor_val12 = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->local_tensor_val12, nx+2, ny+2, nz+2);

  c->local_tensor_val13 = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->local_tensor_val13, nx+2, ny+2, nz+2);

  c->local_tensor_val22 = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->local_tensor_val22, nx+2, ny+2, nz+2);

  c->local_tensor_val23 = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->local_tensor_val23, nx+2, ny+2, nz+2);

  c->local_tensor_val33 = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->local_tensor_val33, nx+2, ny+2, nz+2);

  c->flux_x = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->flux_x, nx+2, ny+2, nz+2);

  c->flux_y = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->flux_y, nx+2, ny+2, nz+2);

  c->flux_z = dallocate_3d(nx+2, ny+2, nz+2);
  dinit_3d(c->flux_z, nx+2, ny+2, nz+2);

  c->grid_x = (double*)calloc(nx+2, sizeof(double));
  c->grid_y = (double*)calloc(ny+2, sizeof(double));
  c->grid_z = (double*)calloc(nz+2, sizeof(double));

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
  double global_max = 0;

  for(i = 0; i < 6; i++)
    tensor_max[i] = 0;

  for(i = 1; i <= c->nz; i++)
  {
    for(j = 1; j <= c->ny; j++)
    {
      for(k = 1; k <= c->nx; k++)
      {
        if(c->local_tensor_val11[i][j][k] > tensor_max[0])
          tensor_max[0] = c->local_tensor_val11[i][j][k];

        if(c->local_tensor_val12[i][j][k] > tensor_max[1])
          tensor_max[1] = c->local_tensor_val12[i][j][k];

        if(c->local_tensor_val13[i][j][k] > tensor_max[2])
          tensor_max[2] = c->local_tensor_val13[i][j][k];

        if(c->local_tensor_val22[i][j][k] > tensor_max[3])
          tensor_max[3] = c->local_tensor_val22[i][j][k];

        if(c->local_tensor_val23[i][j][k] > tensor_max[4])
          tensor_max[4] = c->local_tensor_val23[i][j][k];

        if(c->local_tensor_val33[i][j][k] > tensor_max[5])
          tensor_max[5] = c->local_tensor_val33[i][j][k];
      }
    }
  }

  for(i = 0; i < 6; i++)
  {
    if(tensor_max[i] > max_tensor_value)
      max_tensor_value = tensor_max[i];
  }

  MPI_Allreduce(&max_tensor_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


  step_value = min_double(c->x_step, c->y_step);
  min_step_value = min_double(step_value, c->z_step);

  //dt = 1/(6*max_tensor_value*(min_step_value*min_step_value));
  dt = (min_step_value*min_step_value)/(6*global_max);
  //printf("dt < %f \n", dt);
  return dt;
}

void free3d_global(double ***arr, cube *c)
{
  int i;

  free(arr[0][0]);
  for(i = 0; i <= c->nzc+1; i++)
  {
    free(arr[i]);
  }

  free(arr);
}

void free3d_local(double ***arr, cube *c)
{
  int i;

  free(arr[0][0]);
  for(i = 0; i <= c->nz+1; i++)
  {
    free(arr[i]);
  }

  free(arr);
}

void free1d(double *arr)
{
  free(arr);
}

void enforce_grid_left_right(double ***E, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i <= z+1; i++)
  {
    for(j = 0; j <= y+1; j++)
    {
      for(k = 0; k <= x+1; k+=x+1)
      {
        E[i][j][k] = 0.0;
      }
    }
  }
}

void enforce_grid_up_down(double ***E, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i <= z+1; i++)
  {
    for(j = 0; j <= y+1; j+=y+1)
    {
      for(k = 0; k <= x+1; k++)
      {
        E[i][j][k] = 0.0;
      }
    }
  }
}

void enforce_grid_zup_zdown(double ***E, int x, int y, int z)
{
  int i, j, k;

  for(i = 0; i <= z+1; i+=z+1)
  {
    for(j = 0; j <= y+1; j++)
    {
      for(k = 0; k <= x+1; k++)
      {
        E[i][j][k] = 0.0;
      }
    }
  }
}
