#include "mesh.h"

//void create_tensor_names(cube *c, char *buf11, char *buf12, char *buf13, char *buf22, char *buf23, char *buf33, int rank, int size);
void read_binaryformat(char* filename, double ****matrix, int x, int y, int z);
void write_binaryformat(char* filename, double ***matrix, int x, int y, int z);
