#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ga-mpi/ga.h>
#include <ga-mpi/std_stream.h>
#include "mpi.h"
#include <fstream>
using namespace std;

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

	popsize = mpi_tasks * int((double)popsize/(double)mpi_tasks+0.999);
	printf("GAsize = %d \n",popsize);
	printf("seed = %d \n",seed);
	printf("argc = %x \n",argc);
	printf("argv = %s \n",*argv);
	
	GABin2DecPhenotype map;// tao kieu hinh cho cac bo nhiem sac the// giam chi phi luu tru
	map.add(10, 0.0, 15.0 * M_PI); // su dung 10 bit cho moi bien trong khoang [0,5*PI]x[0,5*PI]
	map.add(10, 0.0, 15.0 * M_PI);
	
	GABin2DecGenome genome(map, objective); // khai bao nhiewm sac the //bo chuyen doi choi nhi phan qua thap phan

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
	ga.mpi_rank(mpi_rank);
	ga.mpi_tasks(mpi_tasks);
	ga.evolve(seed); // xac dinh cot moc muc tieu can vuot qua. seed chinh la moc ban dau de ban vuot qua

	if(mpi_rank == 0)
	{
		genome = ga.statistics().bestIndividual();// tra ve doi tuong thong ke trong thuat toan//tai day tra ve ca the tot nhat
		
		printf("task = %d \n",mpi_tasks);
		
		printf("rank = %d \n",mpi_rank);
		printf("GA result:\n");
		printf("x = %f, y = %f\n",
			genome.phenotype(0), genome.phenotype(1));// anh xa tu so nhi pha qua thap phan// lay kieu hinh duoc chi dinh
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

 	
	std::ofstream output("output.txt", std::ios::app);
  output << genome.phenotype(0) <<"\n";
  output <<  genome.phenotype(1) <<"\n";
  output.close();
	
	error = ((1.-sin(x)*sin(y))+sqrt((x-M_PI*2.5)*(x-M_PI*2.5)+(y-M_PI*2.5)*(y-M_PI*2.5))/10.0)/2.5;
	printf("error = %f \n",error);
	printf("x_temp = %f, y_temp = %f\n",genome.phenotype(0), genome.phenotype(1));
	return error;
}

