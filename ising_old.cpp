#include <random>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <omp.h>

#include "prng_engine.hpp"

using namespace std;

#define PLOT

#include <chrono>

struct timer
{
  static timer timerCost;
  
  static auto now()
  {
    return chrono::high_resolution_clock::now();
  }
  
  size_t tot=0;
  
  size_t n=0;
  
  chrono::high_resolution_clock::time_point from;
  
  void start()
  {
    n++;
    from=now();
  }
  
  void stop()
  {
    tot+=chrono::duration_cast<chrono::nanoseconds>(now()-from).count();
  }
  
  double get(bool sub=true)
  {
    double res=tot;
    if(sub)
      res-=timerCost.tot*(double)n/timerCost.n;
    
    return res/1e9;
  }
};

timer timer::timerCost;

timer totalTime;
// timer computeEnTime;
// timer computeMagTime;
// timer rndGenTime;

int computeEn(vector<int>& conf,int L,int N)
{
  // computeEnTime.start();
  int en=0;
  //#pragma omp parallel for reduction(+:en)
  for(int iSite=0;iSite<N;iSite++)
    {
      int y=iSite/L;
      int x=iSite%L;
      
      int neighSiteX=y*L+(x+1)%L;
      int neighSiteY=((y+1)%L)*L+x;
      
      en-=conf[neighSiteX]*conf[iSite]+conf[neighSiteY]*conf[iSite];
    }
  
      // computeEnTime.stop();
  return en;
}

int computeSiteEn(vector<int>& conf,int L,int N,int iSite)
{
  // computeEnTime.start();
  int en=0;
  int y=iSite/L;
  int x=iSite%L;
  
  int neighSiteX=y*L+(x+1)%L;
  int neighSiteX2=y*L+(x+L-1)%L;
  int neighSiteY=((y+1)%L)*L+x;
  int neighSiteY2=((y+L-1)%L)*L+x;
  
  en-=conf[neighSiteX]*conf[iSite]+conf[neighSiteY]*conf[iSite];
  en-=conf[neighSiteX2]*conf[iSite]+conf[neighSiteY2]*conf[iSite];
  
  // computeEnTime.stop();
  return en;
}

double computeMagnetization(vector<int>& conf,int L,int N)
{
  int mag=0;
  for(int iSite=0;iSite<N;iSite++)
    mag+=conf[iSite];
  
  return (double)mag/N;
}

int main(int narg,char **arg)
{
  if(narg<2)
    {
      cerr<<"Use "<<arg[0]<<" L "<<endl;
      exit(0);
    }
  
#ifdef PLOT
  FILE* gp=popen("gnuplot","w");
  fprintf(gp,"unset key\n");
  fprintf(gp,"set style fill solid\n");
#endif
  
  /** omp_set_num_threads(4); */
  
  cout<<"Maximal number of threads to be used: "<<omp_get_max_threads()<<endl;
  
  double beta=0.9;//.4407228;
  size_t L=atoi(arg[1]);
  size_t N=L*L;
  int seed=124634;
  
  vector<int> conf(N);
  
  // using Gen=sitmo::prng_engine;
  using Gen=mt19937_64;
  vector<Gen> gens(N);
  for(size_t i=0;i<N;i++)
    gens[i].seed(seed+i);
  
  for(size_t i=0;i<N;i++)
    conf[i]=binomial_distribution<int>(1,0.5)(gens[i])*2-1;
  
  int nConfs=1000;
  
  for(size_t i=0;i<10000000;i++)
    {
      timer::timerCost.start();
      timer::timerCost.stop();
    }
  
  totalTime.start();
  
  /** Produce nConfs */
  for(int iConf=0;iConf<nConfs;iConf++)
    {
      /** Update each esite*/
      for(size_t par=0;par<2;par++)
	//#pragma omp parallel for
	for(size_t iSite=0;iSite<N;iSite++)
	  if((iSite%L+iSite/L)%2==par)
	    {
	      // cout<<"Looping on site "<<iSite<<endl; 
	      
	      int backupSiteState=conf[iSite];
	      
	      // cout<<"Before: "<<conf[iSite]<<endl;
	      int enBefore=computeEn(conf,L,N);
	      // cout<<"enBefore: "<<enBefore<<endl;
	      
	      // rndGenTime.start();
	      binomial_distribution siteDistr(1,0.5);
	      if(siteDistr(gens[iSite])==0)
		conf[iSite]=-1;
	      else
		conf[iSite]=+1;
	      // rndGenTime.stop();
	      
	      // cout<<"After: "<<conf[iSite]<<endl;
	      int enAfter=computeEn(conf,L,N);
	      // cout<<"enAfter: "<<enAfter<<endl;
	      
	      int eDiff=enAfter-enBefore;
	      // cout<<"eDiff: "<<eDiff<<endl;
	      
	      if(eDiff<=0)
		// cout<<"Accepted as energy is decreasing"<<endl
		;
	      else
		{
		  double pAcc=exp(-beta*eDiff);
		  
		  // cout<<"Pacc: "<<pAcc<<endl;
		  binomial_distribution<int> distrAcc(1,pAcc);
		  
		  // rndGenTime.start();
		  int acc=distrAcc(gens[iSite]);
		  // cout<<"acc: "<<acc<<endl;
		  // rndGenTime.stop();
		  
		  if(acc==0)
		    {
		      conf[iSite]=backupSiteState;
		      // cout<<"Not accepted"<<endl;
		    }
		  // else
		  // 	cout<<"Accepted"<<endl;
		}
	    }
      
#ifdef PLOT
      fprintf(gp,"plot '-' w boxxyerror\n");
      for(size_t site=0;site<N;site++)
	if(conf[site]==-1)
	  fprintf(gp,"%lg %lg 0.5 0.5\n",site%L+0.5,int(site/L)+0.5);
      fprintf(gp,"e\n");
      fflush(gp);
#endif
      
      // computeMagTime.start();
      // double mag=computeMagnetization(conf,L,N);
      // computeMagTime.stop();
      // cout<<"Mag "<<mag<<endl;
      
      //sleep(1);
    }
  
  totalTime.stop();
  
  double mag=computeMagnetization(conf,L,N);
  cout<<"Mag "<<mag<<endl;
  
  cout<<"TotalTime: "<<totalTime.get()<<" s"<<endl;
  // cout<<"ComputeEnergyTime: "<<computeEnTime.get()<<" s"<<endl;
  // cout<<"ComputeMagnetizationTime: "<<computeMagTime.get()<<" s"<<endl;
  // cout<<"RandomGenTime: "<<rndGenTime.get()<<" s"<<endl;
  cout<<"BenchTime: "<<timer::timerCost.get(false)/timer::timerCost.n<<" s"<<endl;
  
#ifdef PLOT
  pclose(gp);
#endif
  
  return 0;
}
