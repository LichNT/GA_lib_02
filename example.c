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
	printf("popsize = %d \n",popsize);
	printf("seed = %d \n",seed);
	printf("argc = %x \n",argc);
	printf("argv = %s \n",*argv);
	// Create the phenotype for two variables.  The number of bits you can use to
	// represent any number is limited by the type of computer you are using.
	// For this case we use 10 bits for each var, ranging the square domain [0,5*PI]x[0,5*PI]
	GABin2DecPhenotype map;// tao kieu hinh cho cac bo nhiem sac the// giam chi phi luu tru
	map.add(10, 0.0, 5.0 * M_PI);
	map.add(10, 0.0, 5.0 * M_PI);

	// Create the template genome using the phenotype map we just made.
	GABin2DecGenome genome(map, objective); // khai bao nhiewm sac the //bo chuyen doi choi nhi phan qua thap phan

	// Now create the GA using the genome and run it. We'll use sigma truncation
	// scaling so that we can handle negative objective scores.
	GASimpleGA ga(genome); // dan so khong co tinh chong cheo
	//GANoScaling() // khong thay doi so voi muc tieu

	GALinearScaling scaling;// thay doi tuyen tinh so voi ham muc tieu
	ga.minimize();		// by default we want to minimize the objective// thu nho qui mo quan the bang cac chi so 
	ga.populationSize(popsize); // lay kich thuoc quan the -co the thay doi theo thoi gian 
	ga.nGenerations(ngen); // xac dinh luong the he
	ga.pMutation(pmut);// tao dot bien
	ga.pCrossover(pcross); // ^cheo hoa
	ga.scaling(scaling); // tao thu thach vao giet
	if(mpi_rank == 0)
		ga.scoreFilename("evolution.txt"); // ham nay khong hieu luc khi khong co...
	else
		ga.scoreFilename("/dev/null");
	ga.scoreFrequency(1); // ghi lai diem so cua cac the he
	ga.flushFrequency(1); // ghi lai tan suat xoa cua cac the he
	ga.selectScores(GAStatistics::AllScores); // ghi cac diem duoc lua cho vao o dia.

	// Pass MPI data to the GA class
	printf("rankSource = %d \n",mpi_rank);
	ga.mpi_rank(mpi_rank);
	ga.mpi_tasks(mpi_tasks);
	ga.evolve(seed); // xac dinh cot moc muc tieu can vuot qua. seed chinh la moc ban dau de ban vuot qua

	// Dump the GA results to file
	if(mpi_rank == 0)
	{
		genome = ga.statistics().bestIndividual();
		
		printf("task = %d \n",mpi_tasks);
		
		printf("rank = %d \n",mpi_rank);
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

