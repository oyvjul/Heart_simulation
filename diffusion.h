#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <stdio.h>
#include <stdlib.h>

double divergence_cell_direction_x(double ***u, double tensor_x, double tensor_y, double tensor_z, double delta_x, double delta_y, double delta_z, int i, int j, int k);
double divergence_cell_direction_y(double ***u, double tensor_x, double tensor_y, double tensor_z, double delta_x, double delta_y, double delta_z, int i, int j, int k);
double divergence_cell_direction_z(double ***u, double tensor_x, double tensor_y, double tensor_z, double delta_x, double delta_y, double delta_z, int i, int j, int k);

double diffusion(double ***u, double tensor_x, double tensor_x1, double tensor_x2,
  double tensor_y, double tensor_y1, double tensor_y2, double tensor_z, double tensor_z1, double tensor_z2,
  double delta_x, double delta_y, double delta_z, int i, int j, int k);


#endif
