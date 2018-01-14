### A Performance Evaluation of a Structured Heart Simulation ###

### Content information ###

#main.c:
This file is responsible for running the simulation of a diffusion equation sequentially.
Type the number of points you want to use in line 43-45.
and the read files are named in line 72-77.


#mpi_main.c:
This file is responsible for running the simulation of a diffusion equation in Parallel with MPI and optionally OpenMP.
Type the number of points you want to use in line 57-59.
and the read files are named in line 214-219.

#generate.c:
This file generates the coefficients sequentially.
Type the number of points you want to use in line 43-45.
The write files are named in line 102-107.

#generate_mpi.c
This file generates the coefficients in parallel with MPI
Type the number of points you want to use in line 57-59.
The write files are named in line 214-219.

#mesh.h:
This file is responsible for generating the cubic mesh grid.

#io.h:
A header file for reading and writing to binary format.

#tensor.h:
This header is responsible for checking wether is inside or outside the complex geometry.

#diffusion.h:
This file is only an illustrative file (not in use), showing the long road of implementing the diffusion equation.

### Prerequisites ###

To compile the project, you need the following libraries:
MPI
OpenMP (optional)
gcc

### Installing ###

generate the coefficients sequentially:
make gen

generate the coefficients in parallel:
make genmpi

To compile the simulation sequentially:
make seq

To compile the simulation in parallel with MPI:
make mpi

To compile the simulation in parallel with MPI+OpenMP:
make mpih

### Authors ###

-Øyvind Julsrud



