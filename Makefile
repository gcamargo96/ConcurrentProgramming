FILE = GaussJordan_Paralelo
FILES = GaussJordan_Sequencial
NP = 2

SRCDIR = src
BINDIR = bin

par:
	mpicc $(SRCDIR)/$(FILE).c -o $(BINDIR)/$(FILE) -g -Wall -fopenmp

seq:
	gcc $(SRCDIR)/$(FILES).c -o $(BINDIR)/$(FILES) -g -Wall -fopenmp

run:
	mpiexec -np $(NP) $(BINDIR)/$(FILE)

runs:
	./$(BINDIR)/$(FILES)

debug:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(FILE)
 