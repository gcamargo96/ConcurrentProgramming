f = GaussJordan_Paralelo2
all:
	mpicc src/$(f).c -o $(f) -g -Wall -fopenmp
run:
	mpiexec -np 2 $(f)
debug:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(f)
 