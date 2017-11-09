f = GaussJordan_Paralelo
all:
	mpicc src/$(f).c -o $(f) -g -Wall -fopenmp
run:
	./$(f)
debug:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(f)
 