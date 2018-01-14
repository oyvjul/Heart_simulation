#include "diffusion.h"

double cell_direction_x, cell_direction_y, cell_direction_z;
double upper_right_one, lower_right_one, upper_right, lower_right, upper_left_one, lower_left_one, upper_left, lower_left;

double gradient_basis_x(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k, int ii, int jj, int kk)
{
  cell_direction_x = (u[i+1+ii][j+1+jj][k+1+kk] + u[i+1+ii][j+jj][k+1+kk] + u[i+1+ii][j+1+jj][k+kk] + u[i+1+ii][j+jj][k+kk] -
                u[i+ii][j+1+jj][k+1+kk] - u[i+ii][j+jj][k+1+kk] - u[i+ii][j+1+jj][k+kk] - u[i+ii][j+jj][k+kk])/(2*delta_x);

  return cell_direction_x;
  //return (tensor_x*tensor_y*tensor_z)*(cell_direction_x*cell_direction_y*cell_direction_z);
}

double gradient_basis_y(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k, int ii, int jj, int kk)
{
  cell_direction_y = (u[i+1+ii][j+1+jj][k+1+kk] + u[i+ii][j+1+jj][k+1+kk] + u[i+1+ii][j+1+jj][k+kk] + u[i+ii][j+1+jj][k+kk] -
                u[i+1+ii][j+jj][k+1+kk] - u[i+ii][j+jj][k+1+kk] - u[i+1+ii][j+jj][k+kk] - u[i+ii][j+jj][k+kk])/(2*delta_y);

  return cell_direction_y;
  //return (tensor_x*tensor_y*tensor_z)*(cell_direction_x*cell_direction_y*cell_direction_z);
}

double gradient_basis_z(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k, int ii, int jj, int kk)
{
  cell_direction_z = (u[i+1+ii][j+1+jj][k+1+kk] + u[i+1+ii][j+jj][k+1+kk] + u[i+ii][j+1+jj][k+1+kk] + u[i+ii][j+jj][k+1+kk] -
                u[i+1+ii][j+1+jj][k+kk] - u[i+1+ii][j+jj][k+kk] - u[i+ii][j+1+jj][k+kk] - u[i+ii][j+jj][k+kk])/(2*delta_z);

  return cell_direction_z;
  //return (tensor_x*tensor_y*tensor_z)*(cell_direction_x*cell_direction_y*cell_direction_z);
}

double flux_upper_right_z_one_basis(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k, int ii, int jj, int kk)
{
  cell_direction_x = (u[i+1+ii][j+1+jj][k+1+kk] + u[i+1+ii][j+jj][k+1+kk] + u[i+1+ii][j+1+jj][k+kk] + u[i+1+ii][j+jj][k+kk] -
                u[i+ii][j+1+jj][k+1+kk] - u[i+ii][j+jj][k+1+kk] - u[i+ii][j+1+jj][k+kk] - u[i+ii][j+jj][k+kk])/(2*delta_x);

  cell_direction_y = (u[i+1+ii][j+1+jj][k+1+kk] + u[i+ii][j+1+jj][k+1+kk] + u[i+1+ii][j+1+jj][k+kk] + u[i+ii][j+1+jj][k+kk] -
                u[i+1+ii][j+jj][k+1+kk] - u[i+ii][j+jj][k+1+kk] - u[i+1+ii][j+jj][k+kk] - u[i+ii][j+jj][k+kk])/(2*delta_y);

  cell_direction_z = (u[i+1+ii][j+1+jj][k+1+kk] + u[i+1+ii][j+jj][k+1+kk] + u[i+ii][j+1+jj][k+1+kk] + u[i+ii][j+jj][k+1+kk] -
                u[i+1+ii][j+1+jj][k+kk] - u[i+1+ii][j+jj][k+kk] - u[i+ii][j+1+jj][k+kk] - u[i+ii][j+jj][k+kk])/(2*delta_z);

  return ((tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z));
  //return (tensor_x*tensor_y*tensor_z)*(cell_direction_x*cell_direction_y*cell_direction_z);
}

//(i+1/2, j+1/2, k+1/2)
/*double flux_upper_right_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i+1][j+1][k+1] + u[i+1][j][k+1] + u[i+1][j+1][k] + u[i+1][j][k] -
                u[i][j+1][k+1] - u[i][j][k+1] - u[i][j+1][k] - u[i][j][k])/(2*delta_x);

  cell_direction_y = (u[i+1][j+1][k+1] + u[i][j+1][k+1] + u[i+1][j+1][k] + u[i][j+1][k] -
                u[i+1][j][k+1] - u[i][j][k+1] - u[i+1][j][k] - u[i][j][k])/(2*delta_y);

  cell_direction_z = (u[i+1][j+1][k+1] + u[i+1][j][k+1] + u[i][j+1][k+1] + u[i][j][k+1] -
                u[i+1][j+1][k] - u[i+1][j][k] - u[i][j+1][k] - u[i][j][k])/(2*delta_z);

  return ((tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z));
}*/
double flux_upper_right_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, 0);
}

//(i+1/2, j+1/2, k-1/2)//DONE
/*double flux_upper_right(double ***u, double tensor_x, double tensor_y, double tensor_z,
                        double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i+1][j+1][k] + u[i+1][j][k] + u[i+1][j+1][k-1] + u[i+1][j][k-1] -
                u[i][j+1][k] - u[i][j][k] - u[i][j+1][k-1] - u[i][j][k-1])/(2*delta_x);

  cell_direction_y = (u[i+1][j+1][k] + u[i][j+1][k] + u[i+1][j+1][k-1] + u[i][j+1][k-1] -
                u[i+1][j][k] - u[i][j][k] - u[i+1][j][k-1] - u[i][j][k-1])/(2*delta_y);

  cell_direction_z = (u[i+1][j+1][k] + u[i+1][j][k] + u[i][j+1][k] + u[i][j][k] -
                u[i+1][j+1][k-1] - u[i+1][j][k-1] - u[i][j+1][k-1] - u[i][j][k-1])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_upper_right(double ***u, double tensor_x, double tensor_y, double tensor_z,
                        double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, -1);
}

//(i+1/2, j-1/2, k-1/2)
/*double flux_lower_right(double ***u, double tensor_x, double tensor_y, double tensor_z,
                        double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i+1][j][k] + u[i+1][j-1][k] + u[i+1][j][k-1] + u[i+1][j-1][k-1] -
                u[i][j][k] - u[i][j-1][k] - u[i][j][k-1] - u[i][j-1][k-1])/(2*delta_x);

  cell_direction_y = (u[i+1][j][k] + u[i][j][k] + u[i+1][j][k-1] + u[i][j][k-1] -
                u[i+1][j-1][k] - u[i][j-1][k] - u[i+1][j-1][k-1] - u[i][j-1][k-1])/(2*delta_y);

  cell_direction_z = (u[i+1][j][k] + u[i+1][j-1][k] + u[i][j][k] + u[i][j-1][k] -
                u[i+1][j][k-1] - u[i+1][j-1][k-1] - u[i][j][k-1] - u[i][j-1][k-1])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_lower_right(double ***u, double tensor_x, double tensor_y, double tensor_z,
                        double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
    return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, -1);
}

//(i+1/2, j-1/2, k+1/2)
/*double flux_lower_right_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i+1][j][k+1] + u[i+1][j-1][k+1] + u[i+1][j][k] + u[i+1][j-1][k] -
                u[i][j][k+1] - u[i][j-1][k+1] - u[i][j][k] - u[i][j-1][k])/(2*delta_x);

  cell_direction_y = (u[i+1][j][k+1] + u[i][j][k+1] + u[i+1][j][k] + u[i][j][k] -
                u[i+1][j-1][k+1] - u[i][j-1][k+1] - u[i+1][j-1][k] - u[i][j-1][k])/(2*delta_y);

  cell_direction_z = (u[i+1][j][k+1] + u[i+1][j-1][k+1] + u[i][j][k+1] + u[i][j-1][k+1] -
                u[i+1][j][k] - u[i+1][j-1][k] - u[i][j][k] - u[i][j-1][k])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_lower_right_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                            double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, 0);
}

//(i-1/2, j+1/2, k+1/2)
/*double flux_upper_left_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                           double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i][j+1][k+1] + u[i][j][k+1] + u[i][j+1][k] + u[i][j][k] -
                u[i-1][j+1][k+1] - u[i-1][j][k+1] - u[i-1][j+1][k] - u[i-1][j][k])/(2*delta_x);

  cell_direction_y = (u[i][j+1][k+1] + u[i-1][j+1][k+1] + u[i][j+1][k] + u[i-1][j+1][k] -
                u[i][j][k+1] - u[i-1][j][k+1] - u[i][j][k] - u[i-1][j][k])/(2*delta_y);

  cell_direction_z = (u[i][j+1][k+1] + u[i][j][k+1] + u[i-1][j+1][k+1] + u[i-1][j][k+1] -
                u[i][j+1][k] - u[i][j][k] - u[i-1][j+1][k] - u[i-1][j][k])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_upper_left_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                           double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, 0);
}

//(i-1/2, j+1/2, k-1/2)
/*double flux_upper_left(double ***u, double tensor_x, double tensor_y, double tensor_z,
                       double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i][j+1][k] + u[i][j][k] + u[i][j+1][k-1] + u[i][j][k-1] -
                u[i-1][j+1][k] - u[i-1][j][k] - u[i-1][j+1][k-1] - u[i-1][j][k-1])/(2*delta_x);

  cell_direction_y = (u[i][j+1][k] + u[i-1][j+1][k] + u[i][j+1][k-1] + u[i-1][j+1][k-1] -
                u[i][j][k] - u[i-1][j][k] - u[i][j][k-1] - u[i-1][j][k-1])/(2*delta_y);

  cell_direction_z = (u[i][j+1][k] + u[i][j][k] + u[i-1][j+1][k] + u[i-1][j][k] -
                u[i][j+1][k-1] - u[i][j][k-1] - u[i-1][j+1][k-1] - u[i-1][j][k-1])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_upper_left(double ***u, double tensor_x, double tensor_y, double tensor_z,
                       double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, -1);
}

//(i-1/2, j-1/2, k+1/2)
/*double flux_lower_left_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                           double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i][j][k+1] + u[i][j-1][k+1] + u[i][j][k] + u[i][j-1][k] -
                u[i-1][j][k+1] - u[i-1][j-1][k+1] - u[i-1][j][k] - u[i-1][j-1][k])/(2*delta_x);

  cell_direction_y = (u[i][j][k+1] + u[i-1][j][k+1] + u[i][j][k] + u[i-1][j][k] -
                u[i][j-1][k+1] - u[i-1][j-1][k+1] - u[i][j-1][k] - u[i-1][j-1][k])/(2*delta_y);

  cell_direction_z = (u[i][j][k+1] + u[i][j-1][k+1] + u[i-1][j][k+1] + u[i-1][j-1][k+1] -
                u[i][j][k] - u[i][j-1][k] - u[i-1][j][k] - u[i-1][j-1][k])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_lower_left_one(double ***u, double tensor_x, double tensor_y, double tensor_z,
                           double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, 0);
}

//(i-1/2, j-1/2, k-1/2)
/*double flux_lower_left(double ***u, double tensor_x, double tensor_y, double tensor_z,
                       double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  cell_direction_x = (u[i][j][k] + u[i][j-1][k] + u[i][j][k-1] + u[i][j-1][k-1] -
                u[i-1][j][k] - u[i-1][j-1][k] - u[i-1][j][k-1] - u[i-1][j-1][k-1])/(2*delta_x);

  cell_direction_y = (u[i][j][k] + u[i-1][j][k] + u[i][j][k-1] + u[i-1][j][k-1] -
                u[i][j-1][k] - u[i-1][j-1][k] - u[i][j-1][k-1] - u[i-1][j-1][k-1])/(2*delta_y);

  cell_direction_z = (u[i][j][k] + u[i][j-1][k] + u[i-1][j][k] + u[i-1][j-1][k] -
                u[i][j][k-1] - u[i][j-1][k-1] - u[i-1][j][k-1] - u[i-1][j-1][k-1])/(2*delta_z);

  return (tensor_x*cell_direction_x) + (tensor_y*cell_direction_y) + (tensor_z*cell_direction_z);
}*/
double flux_lower_left(double ***u, double tensor_x, double tensor_y, double tensor_z,
                       double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  return flux_upper_right_z_one_basis(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, -1);
}

double diffusion(double ***u, double tensor_x, double tensor_x1, double tensor_x2,
  double tensor_y, double tensor_y1, double tensor_y2, double tensor_z, double tensor_z1, double tensor_z2,
  double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  double upper_right_one_x = (u[i+1][j+1][k+1] + u[i+1][j][k+1] + u[i+1][j+1][k] + u[i+1][j][k] -
                u[i][j+1][k+1] - u[i][j][k+1] - u[i][j+1][k] - u[i][j][k])/(2*delta_x);
  double upper_right_one_y = (u[i+1][j+1][k+1] + u[i][j+1][k+1] + u[i+1][j+1][k] + u[i][j+1][k] -
                u[i+1][j][k+1] - u[i][j][k+1] - u[i+1][j][k] - u[i][j][k])/(2*delta_y);
  double upper_right_one_z = (u[i+1][j+1][k+1] + u[i+1][j][k+1] + u[i][j+1][k+1] + u[i][j][k+1] -
                u[i+1][j+1][k] - u[i+1][j][k] - u[i][j+1][k] - u[i][j][k])/(2*delta_z);

  double lower_right_one_x = (u[i+1][j][k+1] + u[i+1][j-1][k+1] + u[i+1][j][k] + u[i+1][j-1][k] -
                u[i][j][k+1] - u[i][j-1][k+1] - u[i][j][k] - u[i][j-1][k])/(2*delta_x);
  double lower_right_one_y = (u[i+1][j][k+1] + u[i][j][k+1] + u[i+1][j][k] + u[i][j][k] -
                u[i+1][j-1][k+1] - u[i][j-1][k+1] - u[i+1][j-1][k] - u[i][j-1][k])/(2*delta_y);
  double lower_right_one_z = (u[i+1][j][k+1] + u[i+1][j-1][k+1] + u[i][j][k+1] + u[i][j-1][k+1] -
                u[i+1][j][k] - u[i+1][j-1][k] - u[i][j][k] - u[i][j-1][k])/(2*delta_z);

  double upper_right_x = (u[i+1][j+1][k] + u[i+1][j][k] + u[i+1][j+1][k-1] + u[i+1][j][k-1] -
                u[i][j+1][k] - u[i][j][k] - u[i][j+1][k-1] - u[i][j][k-1])/(2*delta_x);
  double upper_right_y = (u[i+1][j+1][k] + u[i][j+1][k] + u[i+1][j+1][k-1] + u[i][j+1][k-1] -
                u[i+1][j][k] - u[i][j][k] - u[i+1][j][k-1] - u[i][j][k-1])/(2*delta_y);
  double upper_right_z = (u[i+1][j+1][k] + u[i+1][j][k] + u[i][j+1][k] + u[i][j][k] -
                u[i+1][j+1][k-1] - u[i+1][j][k-1] - u[i][j+1][k-1] - u[i][j][k-1])/(2*delta_z);

  double lower_right_x = (u[i+1][j][k] + u[i+1][j-1][k] + u[i+1][j][k-1] + u[i+1][j-1][k-1] -
                u[i][j][k] - u[i][j-1][k] - u[i][j][k-1] - u[i][j-1][k-1])/(2*delta_x);
  double lower_right_y = (u[i+1][j][k] + u[i][j][k] + u[i+1][j][k-1] + u[i][j][k-1] -
                u[i+1][j-1][k] - u[i][j-1][k] - u[i+1][j-1][k-1] - u[i][j-1][k-1])/(2*delta_y);
  double lower_right_z = (u[i+1][j][k] + u[i+1][j-1][k] + u[i][j][k] + u[i][j-1][k] -
                u[i+1][j][k-1] - u[i+1][j-1][k-1] - u[i][j][k-1] - u[i][j-1][k-1])/(2*delta_z);

  double upper_left_one_x = (u[i][j+1][k+1] + u[i][j][k+1] + u[i][j+1][k] + u[i][j][k] -
                u[i-1][j+1][k+1] - u[i-1][j][k+1] - u[i-1][j+1][k] - u[i-1][j][k])/(2*delta_x);
  double upper_left_one_y = (u[i][j+1][k+1] + u[i-1][j+1][k+1] + u[i][j+1][k] + u[i-1][j+1][k] -
                u[i][j][k+1] - u[i-1][j][k+1] - u[i][j][k] - u[i-1][j][k])/(2*delta_y);
  double upper_left_one_z = (u[i][j+1][k+1] + u[i][j][k+1] + u[i-1][j+1][k+1] + u[i-1][j][k+1] -
                u[i][j+1][k] - u[i][j][k] - u[i-1][j+1][k] - u[i-1][j][k])/(2*delta_z);

  double lower_left_one_x = (u[i][j][k+1] + u[i][j-1][k+1] + u[i][j][k] + u[i][j-1][k] -
                u[i-1][j][k+1] - u[i-1][j-1][k+1] - u[i-1][j][k] - u[i-1][j-1][k])/(2*delta_x);
  double lower_left_one_y = (u[i][j][k+1] + u[i-1][j][k+1] + u[i][j][k] + u[i-1][j][k] -
                u[i][j-1][k+1] - u[i-1][j-1][k+1] - u[i][j-1][k] - u[i-1][j-1][k])/(2*delta_y);
  double lower_left_one_z = (u[i][j][k+1] + u[i][j-1][k+1] + u[i-1][j][k+1] + u[i-1][j-1][k+1] -
                u[i][j][k] - u[i][j-1][k] - u[i-1][j][k] - u[i-1][j-1][k])/(2*delta_z);

  double upper_left_x = (u[i][j+1][k] + u[i][j][k] + u[i][j+1][k-1] + u[i][j][k-1] -
                u[i-1][j+1][k] - u[i-1][j][k] - u[i-1][j+1][k-1] - u[i-1][j][k-1])/(2*delta_x);
  double upper_left_y = (u[i][j+1][k] + u[i-1][j+1][k] + u[i][j+1][k-1] + u[i-1][j+1][k-1] -
                u[i][j][k] - u[i-1][j][k] - u[i][j][k-1] - u[i-1][j][k-1])/(2*delta_y);
  double upper_left_z = (u[i][j+1][k] + u[i][j][k] + u[i-1][j+1][k] + u[i-1][j][k] -
                u[i][j+1][k-1] - u[i][j][k-1] - u[i-1][j+1][k-1] - u[i-1][j][k-1])/(2*delta_z);

  double lower_left_x = (u[i][j][k] + u[i][j-1][k] + u[i][j][k-1] + u[i][j-1][k-1] -
                u[i-1][j][k] - u[i-1][j-1][k] - u[i-1][j][k-1] - u[i-1][j-1][k-1])/(2*delta_x);
  double lower_left_y = (u[i][j][k] + u[i-1][j][k] + u[i][j][k-1] + u[i-1][j][k-1] -
                u[i][j-1][k] - u[i-1][j-1][k] - u[i][j-1][k-1] - u[i-1][j-1][k-1])/(2*delta_y);
  double lower_left_z = (u[i][j][k] + u[i][j-1][k] + u[i-1][j][k] + u[i-1][j-1][k] -
                u[i][j][k-1] - u[i][j-1][k-1] - u[i-1][j][k-1] - u[i-1][j-1][k-1])/(2*delta_z);

  double divergence_direct_x =
  ((upper_right_one_x*tensor_x + upper_right_one_y*tensor_y + upper_right_one_z*tensor_z)
  + (lower_right_one_x*tensor_x + lower_right_one_y*tensor_y + lower_right_one_z*tensor_z)
  + (upper_right_x*tensor_x + upper_right_y*tensor_y + upper_right_z*tensor_z)
  + (lower_right_x*tensor_x + lower_right_y*tensor_y + lower_right_z*tensor_z)
  - (upper_left_one_x*tensor_x + upper_left_one_y*tensor_y + upper_left_one_z*tensor_z)
  - (lower_left_one_x*tensor_x + lower_left_one_y*tensor_y + lower_left_one_z*tensor_z)
  - (upper_left_x*tensor_x + upper_left_y*tensor_y + upper_left_z*tensor_z)
  - (lower_left_x*tensor_x + lower_left_y*tensor_y + lower_left_z*tensor_z))/(2*delta_x);

  double divergence_direct_y =
  ((upper_right_one_x*tensor_x1 + upper_right_one_y*tensor_y1 + upper_right_one_z*tensor_z1)
  + (upper_left_one_x*tensor_x1 + upper_left_one_y*tensor_y1 + upper_left_one_z*tensor_z1)
  + (upper_right_x*tensor_x1 + upper_right_y*tensor_y1 + upper_right_z*tensor_z1)
  + (upper_left_x*tensor_x1 + upper_left_y*tensor_y1 + upper_left_z*tensor_z1)
  - (lower_right_one_x*tensor_x1 + lower_right_one_y*tensor_y1 + lower_right_one_z*tensor_z1)
  - (lower_left_one_x*tensor_x1 + lower_left_one_y*tensor_y1 + lower_left_one_z*tensor_z1)
  - (lower_right_x*tensor_x1 + lower_right_y*tensor_y1 + lower_right_z*tensor_z1)
  - (lower_left_x*tensor_x1 + lower_left_y*tensor_y1 + lower_left_z*tensor_z1))/(2*delta_y);

  double divergence_direct_z =
  ((upper_right_one_x*tensor_x2 + upper_right_one_y*tensor_y2 + upper_right_one_z*tensor_z2)
  + (lower_right_one_x*tensor_x2 + lower_right_one_y*tensor_y2 + lower_right_one_z*tensor_z2)
  + (upper_left_one_x*tensor_x2 + upper_left_one_y*tensor_y2 + upper_left_one_z*tensor_z2)
  + (lower_left_one_x*tensor_x2 + lower_left_one_y*tensor_y2 + lower_left_one_z*tensor_z2)
  - (upper_right_x*tensor_x2 + upper_right_y*tensor_y2 + upper_right_z*tensor_z2)
  - (lower_right_x*tensor_x2 + lower_right_y*tensor_y2 + lower_right_z*tensor_z2)
  - (upper_left_x*tensor_x2 + upper_left_y*tensor_y2 + upper_left_z*tensor_z2)
  - (lower_left_x*tensor_x2 + lower_left_y*tensor_y2 + lower_left_z*tensor_z2))/(2*delta_z);

  return (divergence_direct_x + divergence_direct_y + divergence_direct_z);
}

double diffusion2(double ***u, double tensor_x, double tensor_x1, double tensor_x2,
  double tensor_y, double tensor_y1, double tensor_y2, double tensor_z, double tensor_z1, double tensor_z2,
  double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  double upper_right_one_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, 0);
  double upper_right_one_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, 0);
  double upper_right_one_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, 0);

  double lower_right_one_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, 0);
  double lower_right_one_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, 0);
  double lower_right_one_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, 0);

  double upper_right_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, -1);
  double upper_right_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, -1);
  double upper_right_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, 0, -1);

  double lower_right_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, -1);
  double lower_right_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, -1);
  double lower_right_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, 0, -1, -1);

  double upper_left_one_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, 0);
  double upper_left_one_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, 0);
  double upper_left_one_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, 0);

  double lower_left_one_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, 0);
  double lower_left_one_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, 0);
  double lower_left_one_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, 0);

  double upper_left_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, -1);
  double upper_left_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, -1);
  double upper_left_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, 0, -1);

  double lower_left_x = gradient_basis_x(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, -1);
  double lower_left_y = gradient_basis_y(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, -1);
  double lower_left_z = gradient_basis_z(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k, -1, -1, -1);

  double divergence_direct_x =
  ((upper_right_one_x*tensor_x + upper_right_one_y*tensor_y + upper_right_one_z*tensor_z)
  + (lower_right_one_x*tensor_x + lower_right_one_y*tensor_y + lower_right_one_z*tensor_z)
  + (upper_right_x*tensor_x + upper_right_y*tensor_y + upper_right_z*tensor_z)
  + (lower_right_x*tensor_x + lower_right_y*tensor_y + lower_right_z*tensor_z)
  - (upper_left_one_x*tensor_x + upper_left_one_y*tensor_y + upper_left_one_z*tensor_z)
  - (lower_left_one_x*tensor_x + lower_left_one_y*tensor_y + lower_left_one_z*tensor_z)
  - (upper_left_x*tensor_x + upper_left_y*tensor_y + upper_left_z*tensor_z)
  - (lower_left_x*tensor_x + lower_left_y*tensor_y + lower_left_z*tensor_z))/(2*delta_x);

  double divergence_direct_y =
  ((upper_right_one_x*tensor_x1 + upper_right_one_y*tensor_y1 + upper_right_one_z*tensor_z1)
  + (upper_left_one_x*tensor_x1 + upper_left_one_y*tensor_y1 + upper_left_one_z*tensor_z1)
  + (upper_right_x*tensor_x1 + upper_right_y*tensor_y1 + upper_right_z*tensor_z1)
  + (upper_left_x*tensor_x1 + upper_left_y*tensor_y1 + upper_left_z*tensor_z1)
  - (lower_right_one_x*tensor_x1 + lower_right_one_y*tensor_y1 + lower_right_one_z*tensor_z1)
  - (lower_left_one_x*tensor_x1 + lower_left_one_y*tensor_y1 + lower_left_one_z*tensor_z1)
  - (lower_right_x*tensor_x1 + lower_right_y*tensor_y1 + lower_right_z*tensor_z1)
  - (lower_left_x*tensor_x1 + lower_left_y*tensor_y1 + lower_left_z*tensor_z1))/(2*delta_y);

  double divergence_direct_z =
  ((upper_right_one_x*tensor_x2 + upper_right_one_y*tensor_y2 + upper_right_one_z*tensor_z2)
  + (lower_right_one_x*tensor_x2 + lower_right_one_y*tensor_y2 + lower_right_one_z*tensor_z2)
  + (upper_left_one_x*tensor_x2 + upper_left_one_y*tensor_y2 + upper_left_one_z*tensor_z2)
  + (lower_left_one_x*tensor_x2 + lower_left_one_y*tensor_y2 + lower_left_one_z*tensor_z2)
  - (upper_right_x*tensor_x2 + upper_right_y*tensor_y2 + upper_right_z*tensor_z2)
  - (lower_right_x*tensor_x2 + lower_right_y*tensor_y2 + lower_right_z*tensor_z2)
  - (upper_left_x*tensor_x2 + upper_left_y*tensor_y2 + upper_left_z*tensor_z2)
  - (lower_left_x*tensor_x2 + lower_left_y*tensor_y2 + lower_left_z*tensor_z2))/(2*delta_z);

  return (divergence_direct_x + divergence_direct_y + divergence_direct_z);
}

double divergence_cell_direction_x(double ***u, double tensor_x, double tensor_y, double tensor_z, double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  upper_right_one = flux_upper_right_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_right_one = flux_lower_right_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_right = flux_upper_right(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_right = flux_lower_right(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_left_one = flux_upper_left_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_left_one = flux_lower_left_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_left = flux_upper_left(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_left = flux_lower_left(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);

  return (upper_right_one + lower_right_one + upper_right + lower_right - upper_left_one - lower_left_one - upper_left - lower_left)/(2*delta_x);
  //return (upper_right_one + lower_right_one + upper_right + lower_right - upper_left_one - lower_left_one - upper_left - lower_left)/(2*((delta_x)*(delta_x + delta_y + delta_z)));
}

double divergence_cell_direction_y(double ***u, double tensor_x, double tensor_y, double tensor_z, double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  upper_right_one = flux_upper_right_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_left_one = flux_upper_left_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_right = flux_upper_right(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_left = flux_upper_left(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_right_one = flux_lower_right_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_left_one = flux_lower_left_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_right =  flux_lower_right(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_left = flux_lower_left(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);

  return (upper_right_one + upper_left_one + upper_right + upper_left - lower_right_one - lower_left_one - lower_right - lower_left)/(2*delta_y);
}

double divergence_cell_direction_z(double ***u, double tensor_x, double tensor_y, double tensor_z, double delta_x, double delta_y, double delta_z, int i, int j, int k)
{
  upper_right_one = flux_upper_right_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_right_one = flux_lower_right_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_left_one = flux_upper_left_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_left_one = flux_lower_left_one(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_right = flux_upper_right(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_right = flux_lower_right(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  upper_left = flux_upper_left(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);
  lower_left = flux_lower_left(u, tensor_x, tensor_y, tensor_z, delta_x, delta_y, delta_z, i, j, k);

  return (upper_right_one + lower_right_one + upper_left_one + lower_left_one - upper_right - lower_right - upper_left - lower_left)/(2*delta_z);
}
