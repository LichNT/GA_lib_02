#include <ga-mpi/GASimpleGA.h>
#include <ga-mpi/garandom.h>


GAParameterList&
GASimpleGA::registerDefaultParameters(GAParameterList& p) {
  GAGeneticAlgorithm::registerDefaultParameters(p);

  p.add(gaNelitism, gaSNelitism,
	GAParameter::BOOLEAN, &gaDefElitism);

  return p;
}

GASimpleGA::GASimpleGA(const GAGenome& c) : GAGeneticAlgorithm(c){
  oldPop = pop->clone();

  el = gaTrue;
  params.add(gaNelitism, gaSNelitism, GAParameter::BOOLEAN, &el);
}
GASimpleGA::GASimpleGA(const GAPopulation& p) : GAGeneticAlgorithm(p){
  oldPop = pop->clone();

  el = gaTrue;
  params.add(gaNelitism, gaSNelitism, GAParameter::BOOLEAN, &el);
}
GASimpleGA::GASimpleGA(const GASimpleGA& ga) : GAGeneticAlgorithm(ga){
  oldPop = (GAPopulation *)0;
  copy(ga);
}
GASimpleGA::~GASimpleGA(){
  delete oldPop;
}
GASimpleGA&
GASimpleGA::operator=(const GASimpleGA& ga){
  if(&ga != this) copy(ga); 
  return *this;
}
void 
GASimpleGA::copy(const GAGeneticAlgorithm & g){
  GAGeneticAlgorithm::copy(g);
  const GASimpleGA& ga = DYN_CAST(const GASimpleGA&,g);
  el = ga.el;
  if(oldPop) oldPop->copy(*(ga.oldPop));
  else oldPop = ga.oldPop->clone();
  oldPop->geneticAlgorithm(*this);
}


int
GASimpleGA::setptr(const char* name, const void* value){
  int status = GAGeneticAlgorithm::setptr(name, value);

  if(strcmp(name, gaNelitism) == 0 ||
     strcmp(name, gaSNelitism) == 0){
    el = (*((int*)value) != 0 ? gaTrue : gaFalse);
    status = 0;
  }
  return status;
}

int
GASimpleGA::get(const char* name, void* value) const {
  int status = GAGeneticAlgorithm::get(name, value);

  if(strcmp(name, gaNelitism) == 0 || 
     strcmp(name, gaSNelitism) == 0){
    *((int*)value) = (el == gaTrue ? 1 : 0);
    status = 0;
  }
  return status;
}

void 
GASimpleGA::objectiveFunction(GAGenome::Evaluator f){
  GAGeneticAlgorithm::objectiveFunction(f);
  for(int i=0; i<pop->size(); i++)
    oldPop->individual(i).evaluator(f);
}

void 
GASimpleGA::objectiveData(const GAEvalData& v){
  GAGeneticAlgorithm::objectiveData(v);
  for(int i=0; i<pop->size(); i++)
    pop->individual(i).evalData(v);
}

const GAPopulation&
GASimpleGA::population(const GAPopulation& p) {
  if(p.size() < 1) {
    GAErr(GA_LOC, className(), "population", gaErrNoIndividuals);
    return *pop;
  }

  GAGeneticAlgorithm::population(p);
  oldPop->copy(*pop->clone());
  oldPop->geneticAlgorithm(*this);

  return *pop;
}

int 
GASimpleGA::populationSize(unsigned int n) {
  GAGeneticAlgorithm::populationSize(n);
  oldPop->size(n);
  return n;
}

int 
GASimpleGA::minimaxi(int m) { 
  GAGeneticAlgorithm::minimaxi(m);
  if(m == MINIMIZE)
    oldPop->order(GAPopulation::LOW_IS_BEST);
  else
    oldPop->order(GAPopulation::HIGH_IS_BEST);
  return minmax;
}


void
GASimpleGA::initialize(unsigned int seed)
{
  GARandomSeed(seed);
  pop->mpi_tasks(vmpi_tasks);
  pop->mpi_rank(vmpi_rank);

  pop->initialize();
  pop->evaluate(gaTrue);	

  stats.reset(*pop);

  if(!scross) 
    GAErr(GA_LOC, className(), "initialize", gaErrNoSexualMating);
}


void
GASimpleGA::step()
{
  int i, mut, c1, c2;
  GAGenome *mom, *dad;          

  GAPopulation *tmppop;		
  tmppop = oldPop;		
  oldPop = pop;			
  pop = tmppop;			

  for(i=0; i<pop->size()-1; i+=2){	
    mom = &(oldPop->select());  
    dad = &(oldPop->select());
    stats.numsel += 2;		

    c1 = c2 = 0;
    if(GAFlipCoin(pCrossover())){
      stats.numcro += (*scross)(*mom, *dad,
				&pop->individual(i), &pop->individual(i+1));
      c1 = c2 = 1;
    }
    else{
      pop->individual( i ).copy(*mom);
      pop->individual(i+1).copy(*dad);
    }
    stats.nummut += (mut = pop->individual( i ).mutate(pMutation()));
    if(mut > 0) c1 = 1;
    stats.nummut += (mut = pop->individual(i+1).mutate(pMutation()));
    if(mut > 0) c2 = 1;

    stats.numeval += c1 + c2;
  }
  if(pop->size() % 2 != 0){	
    mom = &(oldPop->select());  
    dad = &(oldPop->select());
    stats.numsel += 2;		

    c1 = 0;
    if(GAFlipCoin(pCrossover())){
      stats.numcro += (*scross)(*mom, *dad, &pop->individual(i), (GAGenome*)0);
      c1 = 1;
    }
    else{
      if(GARandomBit())
	pop->individual( i ).copy(*mom);
      else
	pop->individual( i ).copy(*dad);
    }
    stats.nummut += (mut = pop->individual( i ).mutate(pMutation()));
    if(mut > 0) c1 = 1;

    stats.numeval += c1;
  }

  pop->mpi_tasks(vmpi_tasks);
  pop->mpi_rank(vmpi_rank);

  stats.numrep += pop->size();
  pop->evaluate(gaTrue);	

  if(minimaxi() == GAGeneticAlgorithm::MAXIMIZE) {
    if(el && oldPop->best().score() > pop->best().score())
      oldPop->replace(pop->replace(&(oldPop->best()), GAPopulation::WORST), 
		      GAPopulation::BEST);
  }
  else {
    if(el && oldPop->best().score() < pop->best().score())
      oldPop->replace(pop->replace(&(oldPop->best()), GAPopulation::WORST), 
		      GAPopulation::BEST);
  }

  stats.update(*pop);		
}
