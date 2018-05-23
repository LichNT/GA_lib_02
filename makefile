include makevars

all: lib RUN_GA

lib:
	cd ga-mpi; $(MAKE)

RUN_GA: RUN_GA.c
	$(CXX) -g -Wall -I. -lm -c RUN_GA.c -o RUN_GA.o
	$(CXX) RUN_GA.o -o RUN_GA -L./ga-mpi -lga-mpi -lm

