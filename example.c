#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ga-mpi/ga.h>
#include <ga-mpi/std_stream.h>
#include "mpi.h"

float objective(GAGenome &);

int mpi_tasks, mpi_rank;

int main(int argc, char **argv)
{
	//Khoi tao MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	

	// khoi tao giong cho quan the
	unsigned int seed = 0;
	for(int i=1 ; i<argc ; i++)
		if(strcmp(argv[i++],"seed") == 0)
			seed = atoi(argv[i]);
	
	// khai bao cac bien moi truong cho quan the
	int popsize  = 100; // so luong quan the
	int ngen     = 100; // so luong the he
	float pmut   = 0.03; // dot bien
	float pcross = 0.65; // ty le chon

	// popsize / mpi_tasks must be an integer
	popsize = mpi_tasks * int((double)popsize/(double)mpi_tasks+0.999);
	printf("popsize :\n");
	printf("size = %popsize \n");
	// Create the phenotype for two variables.  The number of bits you can use to
	// represent any number is limited by the type of computer you are using.
	// For this case we use 10 bits for each var, ranging the square domain [0,5*PI]x[0,5*PI]
	GABin2DecPhenotype map;
	map.add(10, 0.0, 5.0 * M_PI);
	map.add(10, 0.0, 5.0 * M_PI);

	// Create the template genome using the phenotype map we just made.
	GABin2DecGenome genome(map, objective);

	// Now create the GA using the genome and run it. We'll use sigma truncation
	// scaling so that we can handle negative objective scores.
	GASimpleGA ga(genome);
	GALinearScaling scaling;
	ga.minimize();		// by default we want to minimize the objective
	ga.populationSize(popsize);
	ga.nGenerations(ngen);
	ga.pMutation(pmut);
	ga.pCrossover(pcross);
	ga.scaling(scaling);
	if(mpi_rank == 0)
		ga.scoreFilename("evolution.txt");
	else
		ga.scoreFilename("/dev/null");
	ga.scoreFrequency(1);
	ga.flushFrequency(1);
	ga.selectScores(GAStatistics::AllScores);
	// Pass MPI data to the GA class
	ga.mpi_rank(mpi_rank);
	ga.mpi_tasks(mpi_tasks);
	ga.evolve(seed);

	// Dump the GA results to file
	if(mpi_rank == 0)
	{
		genome = ga.statistics().bestIndividual();
		
		printf("GA result:\n");
		printf("x = %f, y = %f\n",
			genome.phenotype(0), genome.phenotype(1));
	}

	MPI_Finalize();

	return 0;
}
 
float objective(GAGenome &c)
{
	GABin2DecGenome &genome = (GABin2DecGenome &)c;
	float x, y, error;

	x = genome.phenotype(0);
	y = genome.phenotype(1);

	// Function with local minima. The lowest is located at (5/2*PI, 5/2*PI)
	error = ((1.-sin(x)*sin(y))+sqrt((x-M_PI*2.5)*(x-M_PI*2.5)+(y-M_PI*2.5)*(y-M_PI*2.5))/10.0)/2.5;

	return error;
}

