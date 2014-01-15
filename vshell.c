#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

struct parstr_hmdsi{
  double alpha;
  double alphabar;
  double beta;
  double q;
  double g;
  double k0;
  double gamma;
  double nu;
  double nuF;
  double nuL;
  double vEP;
  int n;
};

int func_hmdsi(double t, const double y[], double dydt[], void *params){
  struct parstr_hmdsi *ps = (struct parstr_hmdsi *)params;
  double kn;
  double g=ps->g;
  complex zfsum=0;
  complex *phi=(complex *)&y[0];
  complex *dphidt=(complex *)&dydt[0];
  int tid,nthreads;
  int l;

#pragma omp parallel shared (phi,dphidt,ps) private(tid,l,kn) 
  {
#pragma omp for reduction(+:zfsum)
  for(l=1;l<ps->n+1;l++){
    kn=pow(g,l-1)*ps->k0;
    
    dphidt[l]=
    -ps->alphabar*ps->q*kn/(1+kn*kn)*phi[0]*(
	  +( (l<ps->n) ? (g*(1+g*g*kn*kn-ps->q*ps->q)*phi[l+1]) : 0 )
	  -( (l>1) ? ((1+kn*kn/g/g-ps->q*ps->q)*phi[l-1]) : 0 )
     )
    +ps->alpha*pow(kn,4)*(g*g-1)/(1+kn*kn)*(
       +( (l>2 && l<ps->n-1) ? conj(phi[l-2])*conj(phi[l-1])/pow(g,7) : 0 )
       -( (l>2 && l<ps->n-1) ? (g*g+1)/pow(g,3)*conj(phi[l-1])*conj(phi[l+1]) : 0 )
       +( (l>2 && l<ps->n-1) ? pow(g,3)*conj(phi[l+1])*conj(phi[l+2]) : 0 )
    )
      + ((l==20 || l==21) ? ps->gamma: 0 )
    -ps->nu*pow(kn,4)*phi[l]
    -ps->nuL/pow(kn,6)*phi[l];
    zfsum=zfsum + ( (l<ps->n ) ? (ps->alphabar*ps->q*kn*kn*kn*g*(g*g-1)*phi[l]*phi[l+1]) : 0 );
  }
  }
  // the zonal flow equation:
  dphidt[0]=0.0;//zfsum-ps->nuF*(phi[0]-ps->vEP/ps->q);
  //  printf ("zfsum=%f\n",zfsum);
  return GSL_SUCCESS;
}

int main(void){

  gsl_odeiv2_system *sysode=malloc(sizeof(gsl_odeiv2_system));
  struct parstr_hmdsi *pars=malloc(sizeof(struct parstr_hmdsi));
  const gsl_odeiv2_step_type *type=gsl_odeiv2_step_rk8pd;
  gsl_odeiv2_step *step;
  gsl_odeiv2_control *cont;
  gsl_odeiv2_evolve *ev;
  int N;
  double dtout=1.0e-1;
  double told=-1e4;
  double t=0.0;
  double tmax=1.0e2;
  double h=1.0e-12;
  double *y;
  int l;
  int status;
  FILE *fout;
  complex *phi;
  double kn;
  pars->alphabar=0.0;
  pars->beta=0.0;
  pars->g=0.5*(1.0+sqrt(5.0));
  pars->alpha=pars->g*pars->g;
  pars->q=0.0;
  pars->k0=1.0;
  pars->gamma=0.01;
  pars->nu=1.0e-28;
  pars->nuF=0.0;
  pars->nuL=1.0e3;
  pars->vEP=0.0;
  pars->n=40; // +zf
  N=(pars->n+1);
  y=malloc(sizeof(complex)*N);
  phi=(complex *)&y[0];
  phi[0]=0.0;
  for(l=1;l<N;l++){
    kn=pow(pars->g,l)*pars->k0;
    phi[l]=1e-12*exp(-4*pow(pow(pars->g/1.8,l)-pow(pars->g/1.8,3),2))*exp(I*(rand()*1.0/RAND_MAX)*2.0*M_PI);
    //    phi[l]=1e0*pow(kn,-1)*exp(I*(rand()*1.0/RAND_MAX)*2.0*M_PI);
  }

  sysode->function=func_hmdsi;
  sysode->jacobian=NULL;
  sysode->dimension=N*sizeof(complex)/sizeof(double);
  sysode->params=pars;

  step=gsl_odeiv2_step_alloc(type,sysode->dimension);
  //cont=gsl_odeiv2_control_y_new (1e-20, 0.0);
  cont=gsl_odeiv2_control_standard_new (1.0e-12, 1.0e-4, 1.0,0.0);
  ev=gsl_odeiv2_evolve_alloc(sysode->dimension);

  fout=fopen("vshell_out.dat","w+");
  fclose(fout);
  printf("\n");
  while(t<tmax){
    fout=fopen("vshell_out.dat","a");
    printf("\rt=%e",t);
    if((t-told)>=dtout){
      printf("t=%e\n",t);
      fprintf(fout,"\n\n%e %e\n",pars->q,cabs(phi[0]));
      for(l=1;l<N;l++)
	fprintf(fout,"%e %e\n",pars->k0*pow(pars->g,l),cabs(phi[l]));
      told=t;
    }
    fclose(fout);
    status=gsl_odeiv2_evolve_apply(ev,cont,step,sysode,&t,tmax,&h,y);
    //    status=gsl_odeiv2_evolve_apply_fixed_step (ev,cont,step,sysode,&t,h,y);
    if (status != GSL_SUCCESS)
      break;
  }
}
