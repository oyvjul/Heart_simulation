COMPILER_CC = gcc-7
COMPILER_MPI = mpicc
MPI_CC = OMPI_CC=gcc-7
CC_FILES = diffusion.c tensor.c io.c mesh.c
CC_FLAGS = -fopenmp

seq: main.c 
	  ${COMPILER_CC} ${CC_FLAGS} main.c ${CC_FILES} -DTEST -o main
seqb: main.c 
	  ${COMPILER_CC} ${CC_FLAGS} main.c ${CC_FILES} -DBLOCK -o main
seqn: main.c 
	  ${COMPILER_CC} ${CC_FLAGS} main.c ${CC_FILES} -DNEW -o main
gen: main.c 
	  ${COMPILER_CC} ${CC_FLAGS} generate.c ${CC_FILES} -o generate

mpi: mpi_main.c
	  ${MPI_CC} ${COMPILER_MPI} ${CC_FLAGS} mpi_main.c ${CC_FILES} -DMPI -o mpi_main
mpih: mpi_main.c
	  ${MPI_CC} ${COMPILER_MPI} ${CC_FLAGS} mpi_main.c ${CC_FILES} -DHYBRID -o mpi_main
genmpi: mpi_main.c
	  ${MPI_CC} ${COMPILER_MPI} ${CC_FLAGS} generate_mpi.c ${CC_FILES} -o generate_mpi

mpir: mpi_main
	  mpirun -np 8 ./mpi_main
genmpir: generate_mpi
	  mpirun -np 8 ./generate_mpi
seqr: main
	  ./main
genr: generate
	  ./generate
clean: 
	  @rm mpi_main main generate_mpi generate