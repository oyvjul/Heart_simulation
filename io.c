#include <stdio.h>
#include "io.h"
#include "mesh.h"

/*void create_tensor_names(cube *c, char *buf11, char *buf12, char *buf13, char *buf22, char *buf23, char *buf33, int rank, int size)
{
  char str_rank[12];
  char str_mesh_size[12];
  char str_size[12];
  sprintf(str_rank, "%d", rank);
  sprintf(str_size, "%d", size);
  sprintf(str_mesh_size, "%dx%dx%d", c->nxc, c->nyc, c->nzc);
  char *str_name = "mpi_mesh/local_tensor_val11";
  char *str_name11 = "mpi_mesh/local_tensor_val11";
  char *str_name12 = "mpi_mesh/local_tensor_val12";
  char *str_name13 = "mpi_mesh/local_tensor_val13";
  char *str_name22 = "mpi_mesh/local_tensor_val22";
  char *str_name23 = "mpi_mesh/local_tensor_val23";
  char *str_name33 = "mpi_mesh/local_tensor_val33";
  char *str_end = ".tensor";
  snprintf(*buf11, sizeof *buf11, "%s_%s_%s_%s%s", str_name11, str_mesh_size, str_rank, str_size, str_end);
  snprintf(*buf12, sizeof *buf12, "%s_%s_%s_%s%s", str_name12, str_mesh_size, str_rank, str_size, str_end);
  snprintf(*buf13, sizeof *buf13, "%s_%s_%s_%s%s", str_name13, str_mesh_size, str_rank, str_size, str_end);
  snprintf(*buf22, sizeof *buf22, "%s_%s_%s_%s%s", str_name22, str_mesh_size, str_rank, str_size, str_end);
  snprintf(*buf23, sizeof *buf23, "%s_%s_%s_%s%s", str_name23, str_mesh_size, str_rank, str_size, str_end);
  snprintf(*buf33, sizeof *buf33, "%s_%s_%s_%s%s", str_name33, str_mesh_size, str_rank, str_size, str_end);
}*/

void read_binaryformat(char* filename, double ****matrix, int x, int y, int z)
{
    int i;
    FILE* fp = fopen (filename,"rb");
    /*fread (x, sizeof(int), 1, fp);
    fread (y, sizeof(int), 1, fp);
    fread (z, sizeof(int), 1, fp);*/
    fread ((*matrix)[0][0], sizeof(double), x*y*z, fp);
    fclose (fp);
}

void write_binaryformat(char* filename, double ***matrix, int x, int y, int z)
{
    FILE *fp = fopen (filename,"wb");
    /*fwrite (&x, sizeof(int), 1, fp);
    fwrite (&y, sizeof(int), 1, fp);
    fwrite (&z, sizeof(int), 1, fp);*/
    fwrite (matrix[0][0], sizeof(double), x*y*z, fp);
    fclose (fp);
}
