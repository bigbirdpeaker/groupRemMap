
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

//////////// c code for group remmap method ////////////
//////////// using shooting methods  ////////////

///////////////////////////////////////
////// statements of affinity functions
///////////////////////////////////////


double SoftShrink(double y, double lam);

// calculate theta for each group:
void cal_theta(double *W,double gamma,double tau,int *G,int *G_label,int J,int P,int Q,double *Beta,double *theta);

// calculate l2 penalty parameter vector for weigthed remmap:
void cal_lam2(double *theta,double *W,double gamma,int *G,int *G_label,int J,int P,double *lambda2);

void Assign(int P, int Q, double *data_old, double *data_copy);
void CalBnorm(int P, int Q, double *Beta, int *C_m, double *Bnorm);
void Update(int cur_p, int N, int P, int Q, double lambda1, double lambda2, int *C_m, double *X_m, double *Xnorm, double *E, double *Beta, double *Bnorm, double *phi_old, double *phi);
double Dist(int P, int Q, double *phi_last, double * phi);

void MultiRegGroupLasso (int *NN, int *PP, int *QQ, double *X_m, double *Y_m, int *C_m, double *lam1,  double *lam2,
                  double *Phi_output, double *Phi_debug, int *N_iter, double *RSS, double *E_debug);


//////////// main functions:
void groupremmap(int *NN, int *PP, int *QQ, double *X_m, double *Y_m, int *C_m,int *G, int *JJ,int * G_label,
				 double * W,  double *lam1, double *Tau,double *Gamma, double *Phi_output,
				 double *Phi_debug, int *N_iter, double *RSS, double *E_debug);

void groupremmapini(int *NN, int *PP, int *QQ, double *X_m, double *Y_m, int *C_m,int *G, int *JJ,int * G_label,
				 double * W,  double *lam1, double *Tau,double *Gamma, double *Phi_initial,double *Phi_output,
				 double *Phi_debug, int *N_iter, double *RSS, double *E_debug);


//////////////////////////////////////////////////////
////////////////////////// no initial value

void groupremmap(int *NN, int *PP, int *QQ, double *X_m, double *Y_m, int *C_m,int *G, int *JJ,int * G_label,
				 double * W,  double *lam1, double *Tau,double *Gamma, double *Phi_output,
				 double *Phi_debug, int *N_iter, double *RSS, double *E_debug)
{
	/// Variables:
	/// N: Number of observations;
	/// P: Number of predictors;
	/// Q: Number of outcomes;

	/// X_m: predictor matrix of N by P; each column has mean 0;
	/// Y_m: outcome matrix of N by Q; each column has mean 0;
	/// C_m: matrix of P by Q, input data; indicator for "self relationship";
     	///   C_m[i,j]=1 means the corresponding beta[i,j] will be penalized;	
        ///   C_m[i,j]=2 means the corresponding beta[i,j] will not be penalized.
        ///   C_m[i,j]=0 means the corresponding beta[i,j] will not be considered in the model.

	/// G: group indicator of P by 1;
    /// J: total group number;
	/// G_label: group lable of J by 1;
	/// W: group weight of J by 1;

    

	/// lam1: l1 penalty parameter;
	/// tau: bridge penalty parameter;
	/// gamma: bridge degree(0<gamma<1);

	/// Phi_output: matrix of P by Q, estimated coefficients.
	/// Phi_debug: matrix of P by Q, for debug
	/// N_iter:   indicate the total iteration
	/// RSS: the sum of squre error

	int N, P, Q,J;
	int n,p,q;
    long int n_iter0;


	double temp;
	double lambda1,tau,gamma;
	double *Xnorm0;
	double *Beta0;
	double *phi_first;
	double *theta;
	double *lam2;
	double eps0;
	double flag0;

	N=*NN;
	P=*PP;
	Q=*QQ;
	J=*JJ;
	lambda1=*lam1;
	tau=*Tau;
	gamma=*Gamma;

    n_iter0=0;
	eps0=1e-6;

	Xnorm0=(double *) malloc (P*sizeof(double));
	Beta0=(double *) malloc (P*Q*sizeof(double));
	phi_first=(double *) malloc (P*Q*sizeof(double));
	theta=(double *) malloc (J*sizeof(double));
	lam2=(double *) malloc (P*sizeof(double));



	/////////////////////////////////////////////////////
	////////// step 0. get initial values of coefficients
	/////////////////////////////////////////////////////

    //////// (1). calculate Xnorm: 
	for(p=0;p<P;p++)	
	{
		 Xnorm0[p]=0;
		 for(n=0;n<N;n++) Xnorm0[p]=Xnorm0[p]+X_m[n*P+p]*X_m[n*P+p];	      
	 }

    //////// (2) calculate lasso solution Beta:
	for(p=0;p<P;p++)
	   for(q=0;q<Q;q++)
	   {
         
            if(C_m[p*Q+q]==0) /// not considered in the model
               Beta0[p*Q+q]=0;
            else
            {
		       temp=0;
		       for(n=0;n<N;n++)
		           temp=temp+X_m[n*P+p]*Y_m[n*Q+q];

		       if(C_m[p*Q+q]==1) /// penalized
				    Beta0[p*Q+q]=SoftShrink(temp, lambda1)/Xnorm0[p];
               else              /// not penalized
                    Beta0[p*Q+q]=temp/Xnorm0[p];
            }
	  }
   ///////// (3) treat Beta0 as initial value;
     Assign(P, Q, Beta0, phi_first);



	////////////////////////////////////////////////////////////////
	////////// step 1. start to update coefficients with group remmap
	/////////////////////////////////////////////////////////////////
   
	flag0=100;
	while(flag0>eps0 && n_iter0<1e+4)
	{
	    ////////// (1). calculate theta:
	    
          cal_theta(W,gamma,tau,G,G_label,J,P,Q,phi_first,theta);
	
	    ////////// (2). update coefficient with weighted remmap method:

		 //////// (2.1) calculate lambda 2:
		  
		  cal_lam2(theta,W,gamma,G,G_label,J,P,lam2);

		 //////// (2.2) apply weighted remmap to update coefficients:

          MultiRegGroupLasso (NN, PP, QQ, X_m, Y_m, C_m, lam1, lam2,Phi_output, Phi_debug, N_iter, RSS, E_debug);

        ///////// (2.3) calculate the dist between phi_first and phi_output:

		  flag0=Dist(P, Q, Phi_output,phi_first);


          Assign(P, Q, Phi_output, phi_first);

		  n_iter0=n_iter0+1;

	}   /// end of "while(flag0>eps0 && n_iter0<1e+10)"


    //////// free allocated variables
     *N_iter=n_iter0;
      free(Xnorm0);
	  free(Beta0);
	  free(phi_first);
	  free(theta);
      free(lam2);
  
} /// end of groupremmap function


//////////////////////////////////////////////////////
////////////////////////// initial value

void groupremmapIni(int *NN, int *PP, int *QQ, double *X_m, double *Y_m, int *C_m,int *G, int *JJ,int * G_label,
				 double * W,  double *lam1, double *Tau,double *Gamma, double *Phi_initial,double *Phi_output,
				 double *Phi_debug, int *N_iter, double *RSS, double *E_debug)
{
	/// Variables:
	/// N: Number of observations;
	/// P: Number of DNA;
	/// Q: Number of mRNA;

	/// X_m: predictor matrix of N by P; each column has mean 0.
	/// Y_m: outcome matrix of N by Q; each column has mean 0.
	/// C_m: matrix of P by Q, input data; indicator for "self relationship";
     	///   C_m[i,j]=1 means the corresponding beta[i,j] will be penalized;	
        ///   C_m[i,j]=2 means the corresponding beta[i,j] will not be penalized.
        ///   C_m[i,j]=0 means the corresponding beta[i,j] will not be considered in the model.

	/// G: group indicator of P by 1;
    /// JJ: total group number;
	/// G_label: group lable of JJ by 1;
	/// W: group weight of JJ by 1;

    

	/// lam1: l1 penalty parameter;
	/// tau: bridge penalty parameter;
	/// gamma: bridge degree(0<gamma<1);

	/// Phi_output: matrix of P by Q, estimated coefficients.
	/// Phi_debug: matrix of P by Q, for debug
	/// N_iter:   indicate the total iteration
	/// RSS: the sum of squre error

	int N, P, Q,J;
	long int n_iter0;


	double lambda1,tau,gamma;
	double *phi_first;
	double *theta;
	double *lam2;
	double eps0;
	double flag0;

	N=*NN;
	P=*PP;
	Q=*QQ;
	J=*JJ;
	lambda1=*lam1;
	tau=*Tau;
	gamma=*Gamma;



    n_iter0=0;
	//eps0=1e-6;
    eps0=1e-6;

	phi_first=(double *) malloc (P*Q*sizeof(double));
	theta=(double *) malloc (J*sizeof(double));
	lam2=(double *) malloc (P*sizeof(double));


	/////////////////////////////////////////////////////
	////////// step 0. get initial values of coefficients
	/////////////////////////////////////////////////////

     Assign(P, Q, Phi_initial, phi_first);

	////////////////////////////////////////////////////////////////
	////////// step 1. start to update coefficients with group remmap
	/////////////////////////////////////////////////////////////////
   
	flag0=100;
	while(flag0>eps0 && n_iter0<1e+4)
	{
	    ////////// (1). calculate theta:
	    
          cal_theta(W,gamma,tau,G,G_label,J,P,Q,phi_first,theta);
	
	    ////////// (2). update coefficient with weighted remmap method:

		 //////// (2.1) calculate lambda 2:
		  
		  cal_lam2(theta,W,gamma,G,G_label,J,P,lam2);

		 //////// (2.2) apply weighted remmap to update coefficients:

          MultiRegGroupLasso (NN, PP, QQ, X_m, Y_m, C_m, lam1, lam2,Phi_output, Phi_debug, N_iter, RSS, E_debug);

        ///////// (2.3) calculate the dist between phi_first and phi_output:

		  flag0=Dist(P, Q, Phi_output,phi_first);

          if (flag0>eps0)
		  {
          Assign(P, Q, Phi_output, phi_first);

		  n_iter0=n_iter0+1;
		  }
	
	}   /// end of "while(flag0>eps0 && n_iter0<1e+10)"


    //////// free allocated variables

      *N_iter=n_iter0;
	  free(phi_first);
	  free(theta);
      free(lam2);
  
} /// end of groupremmapini function








///////////////////////////////////////
//////  affinity functions
///////////////////////////////////////


///////////////////////// weighted remmap solution:

void MultiRegGroupLasso (int *NN, int *PP, int *QQ, double *X_m, double *Y_m, int *C_m, double *lam1, double *lam2,
                         double *Phi_output, double *Phi_debug, int *N_iter, double *RSS, double *E_debug)
{
	/// Variables:
	/// N: Number of observations;
	/// P: Number of DNA;
	/// Q: Number of mRNA;

	/// X_m: predictor matrix of N by P; each column has mean 0.
	/// Y_m: outcome matrix of N by Q; each column has mean 0.
	/// C_m: matrix of P by Q, input data; indicator for "self relationship";
     	///   C_m[i,j]=1 means the corresponding beta[i,j] will be penalized;	
        ///   C_m[i,j]=2 means the corresponding beta[i,j] will not be penalized.
        ///   C_m[i,j]=0 means the corresponding beta[i,j] will not be considered in the model.

	/// lambda1: l1 penalty tuning parameter;
	/// lambda2: l2 penalty tuning parameter vector of P by 1;

	/// Phi_output: matrix of P by Q, estimated coefficients.
	/// Phi_debug: matrix of P by Q, for debug
	/// N_iter:   indicate the total iteration
	/// RSS: the sum of squre error

	int N, P, Q;
	int n,p,q;
	int i;
	long int n_iter,n_iter_a;
	int cur_p;
	int *pick;
	int n_pick;
	int *unpick;
	int n_unpick;

    double rss;
	double temp,temp1;
	double lambda1,lambda2;
	double *Xnorm;
	double *Beta;
	double *phi;
	double *phi_old, *phi_last;
	double *E;
	double *Bnorm;
	double eps;
	double flag;
	double flag_a;

	lambda1=*lam1;
	N=*NN;
	P=*PP;
	Q=*QQ;

    n_iter=0;

	eps=1e-6;

	pick=(int *) malloc (P*sizeof(int));
	unpick=(int *) malloc (P*sizeof(int));
	Xnorm=(double *) malloc (P*sizeof(double));
	Beta=(double *) malloc (P*Q*sizeof(double));
	phi=(double *) malloc (P*Q*sizeof(double));
	phi_old=(double *) malloc (P*Q*sizeof(double));
	phi_last=(double *) malloc (P*Q*sizeof(double));
	E=(double *) malloc (N*Q*sizeof(double));
	Bnorm=(double *) malloc (P*sizeof(double));

///// initial value
	for(p=0;p<P;p++)
	 {
		 Xnorm[p]=0;
		 for(n=0;n<N;n++)
		   Xnorm[p]=Xnorm[p]+X_m[n*P+p]*X_m[n*P+p];
	 }

////////(1) calculate lasso solution Beta
	for(p=0;p<P;p++)
	 for(q=0;q<Q;q++)
	  {
         
            if(C_m[p*Q+q]==0) /// not considered in the model
               Beta[p*Q+q]=0;
            else
            {
		  temp=0;
		  for(n=0;n<N;n++)
		    temp=temp+X_m[n*P+p]*Y_m[n*Q+q];
		  if(C_m[p*Q+q]==1) /// penalized
				Beta[p*Q+q]=SoftShrink(temp, lambda1)/Xnorm[p];
                   else /// not penalized
                                Beta[p*Q+q]=temp/Xnorm[p];
            }
	  }
 ///////// (2) calculate Bnorm;
    CalBnorm(P, Q, Beta, C_m, Bnorm);

 //////// (3) calculate phi

    for(p=0;p<P;p++)
	{
	   lambda2=lam2[p];
       for(q=0;q<Q;q++)
         {            
            if(C_m[p*Q+q]==0) /// not considered in the model
               phi[p*Q+q]=0;
            else if(C_m[p*Q+q]==1 && Bnorm[p]>eps) /// penalized			  
                  {
			         temp=Xnorm[p]*Bnorm[p];
			         temp1=SoftShrink(temp, lambda2);
    		         phi[p*Q+q]=temp1*Beta[p*Q+q]/temp;
		       }            
			   else
	              	 phi[p*Q+q]=Beta[p*Q+q]; /// not penalized
         }

	}

////////// (4) Derive active set
     CalBnorm(P, Q, phi, C_m, Bnorm);
///////// (5) Residue
    for(n=0;n<N;n++)
     for(q=0;q<Q;q++)
      {
		  temp=0;
		  for(p=0;p<P;p++)
		    temp=temp+phi[p*Q+q]*X_m[n*P+p];
 		  E[n*Q+q]=Y_m[n*Q+q]-temp;
      }
        /////////////////////////////////////////////////
        ///////////// (6) begin update
        ////////////////////////////////////////////////

             flag=100;
             while(flag>eps && n_iter<1e+5)
             {
			   //printf("iter= %d", n_iter);

			   //////////// derive active set
			   n_pick = 0;
			   n_unpick=0;
			   for(p = 0; p<P;p++)
			    {
			      if( Bnorm[p]>eps)
				  {
				 	 pick[n_pick] =p;
				 	 n_pick = n_pick + 1;
		           }
		           else
		           {
					 unpick[n_unpick] =p;
				 	 n_unpick = n_unpick + 1;
				   }
			     }

			   //////////// prepare for active set
			   flag_a=100;
               n_iter_a=0;

			   while(flag_a>eps && n_iter_a<1e+5)/////////////////1) Active set
			    {

				 ///phi_last=phi
				 Assign(P, Q, phi, phi_last);				 
                 Assign(P, Q, phi, phi_old); 
				 
                 /// update coefficients that are penalized                 
                 for(i=0;i<n_pick;i++)
				 {	
                     cur_p=pick[i];
				     lambda2=lam2[cur_p];

        		     Update(cur_p, N, P, Q, lambda1, lambda2, C_m, X_m, Xnorm, E, Beta, Bnorm, phi_old, phi);
			         n_iter_a=n_iter_a+1;				  
				 }

				  /// update coefficients that are not penalized
				  for(i=0; i<n_unpick; i++)
				  {
					  p=unpick[i];
					  for(q=0; q<Q; q++)
						   if(C_m[p*Q+q]==2)
						    {
								temp=0;
								for(n=0;n<N;n++)
								  temp=temp+E[n*Q+q]*X_m[n*P+p];
								phi[p*Q+q]=temp/Xnorm[p]+phi_old[p*Q+q];	
                                                ///update residue
								for(n=0; n<N;n++)
								     E[n*Q+q]=E[n*Q+q]+(phi_old[p*Q+q]-phi[p*Q+q])*X_m[n*P+p];
								///update phi_old
								phi_old[p*Q+q]=phi[p*Q+q];                 
                                n_iter_a=n_iter_a+1;
					         } /// end if
				  }              
                         flag_a=Dist(P, Q, phi_last, phi);
			    }///end of active set while(flag.a>1e-6)
               //////////////// 2) Full loop
               Assign(P, Q, phi, phi_last);
               Assign(P, Q, phi, phi_old);
               for(p=0;p<P;p++)               
                {
				   lambda2=lam2[p];
		           Update(p, N, P, Q, lambda1, lambda2, C_m, X_m, Xnorm, E, Beta, Bnorm, phi_old, phi);	
                   n_iter=n_iter+1;	
            	}	
       	flag=Dist(P, Q, phi_last, phi);	
            }//end of all loop while(flag>1e-6)

  //////////// calculate Residue
    rss=0;  
    for(n=0;n<N;n++) 
     for(q=0;q<Q;q++)
	   {
		  rss=rss+E[n*Q+q]*E[n*Q+q];
	   }
    Assign(N, Q, E, E_debug);

//////////// return phi
Assign(P, Q, phi, Phi_output);
Assign(P, Q, phi_last, Phi_debug);
*N_iter=n_iter;
*RSS=rss;

//////// free allocated variables
free(pick);
free(unpick);
free(Xnorm);
free(Beta);
free(phi);
free(phi_old);
free(phi_last);
free(E);
free(Bnorm);
}///end MultiRegGroupLasso function 


///////////////// ////////////////////////////
double SoftShrink(double y, double lam)
{
	double temp;
	double result;
	if(y>0)
	 temp=y-lam;
	else
	 temp=-y-lam;
	if(temp<=0)
	 result=0;
	else
	 {
		 result=temp;
		 if(y<0)
		  result=temp*(-1);
	  }
	return(result);
}

///////////////
void cal_theta(double *W,double gamma,double tau,int *G,int *G_label,int J,int P,int Q,double *Beta,double *theta)
{
     int p,q,j;
	 double group_norm,l2_norm=0;
	 for (j=0;j<J;j++)
	 {
	     group_norm=0;
		 for (p=0;p<P;p++) 
		 {
		      if (G[p]==G_label[j]) 
			  {
				 l2_norm=0;
			     for (q=0;q<Q;q++) 
					 l2_norm=l2_norm+Beta[p*Q+q]*Beta[p*Q+q];			 

				 l2_norm=sqrt(l2_norm);
			  }
			  group_norm=group_norm+l2_norm;
		 }

		 theta[j]=W[j]*pow((1-gamma)/(tau*gamma),gamma)*pow(group_norm,gamma);
	 }


}

///////////////
void cal_lam2(double *theta,double *W,double gamma,int *G,int *G_label,int J,int P,double *lambda2)
{
    int p,j;

	for(p=0;p<P;p++)
		for(j=0;j<J;j++)
		{
		   if(G[p]==G_label[j]) 
                    {
                      if (theta[j]!=0) lambda2[p]=pow(theta[j],(gamma-1)/gamma)*pow(W[j],1/gamma);
                      else             lambda2[p]=100000;
                
                     }
		}

}


//////////////////
void CalBnorm(int P, int Q, double *Beta, int *C_m, double *Bnorm)
{
	int p,q;
    for(p=0;p<P;p++)
     {
		 Bnorm[p]=0;
         for(q=0;q<Q;q++)
         {
		   if(C_m[p*Q+q]==1)
            Bnorm[p]=Bnorm[p]+Beta[p*Q+q]*Beta[p*Q+q];
		 }
         Bnorm[p]=sqrt(Bnorm[p]);
     }
}

//////////////////
void Assign(int P, int Q, double *data_old, double *data_copy)
{
	int p, q;

	for(p=0;p<P;p++)
	 for(q=0;q<Q;q++)
	  data_copy[p*Q+q]=data_old[p*Q+q];
}

///////////////////
void Update(int cur_p, int N, int P, int Q, double lambda1, double lambda2, int *C_m, double *X_m, double *Xnorm, double *E, double *Beta, double *Bnorm, double *phi_old, double *phi)
{
	int n, p, q;
	double temp, temp1;
	p=cur_p;	
      ////// calculate lasso solution Beta
	for(q=0;q<Q;q++)
	 {
       if(C_m[p*Q+q]==0)
            Beta[p*Q+q]=0;
       else
       {
	    temp=0;
	    for(n=0;n<N;n++)
	      temp=temp+E[n*Q+q]*X_m[n*P+p];
    	temp1=temp+phi_old[p*Q+q]*Xnorm[p];
	    if(C_m[p*Q+q]==1)
	      Beta[p*Q+q]=SoftShrink(temp1, lambda1)/Xnorm[p];
	    else
		  Beta[p*Q+q]=temp1/Xnorm[p];
       }
	 }    
    ////// calculate norm of lasso solution Beta    
    Bnorm[p]=0;
    for(q=0;q<Q;q++)
    {
	  if(C_m[p*Q+q]==1)
        Bnorm[p]=Bnorm[p]+Beta[p*Q+q]*Beta[p*Q+q];
    }
    Bnorm[p]=sqrt(Bnorm[p]);
    ////// update phi    
    for(q=0;q<Q;q++)
    {
      if(C_m[p*Q+q]==0)  phi[p*Q+q]=0;

	  else if(C_m[p*Q+q]==1 && Bnorm[p]>1e-6)
	   {
		temp=Xnorm[p]*Bnorm[p];
	    phi[p*Q+q]=Beta[p*Q+q]*SoftShrink(temp, lambda2)/temp;
	   }
	  else
	    phi[p*Q+q]=Beta[p*Q+q];
	}    
    ////// update residue    
    for(q=0;q<Q;q++)
      for(n=0;n<N;n++)
       {
		E[n*Q+q]=E[n*Q+q]+(phi_old[p*Q+q]-phi[p*Q+q])*X_m[n*P+p];
	   }
   /////// update phi_old    
   for(q=0;q<Q;q++)
      phi_old[p*Q+q]=phi[p*Q+q];
   /////// update Bnorm
    Bnorm[p]=0;
	    for(q=0;q<Q;q++)
	    {
		  if(C_m[p*Q+q]==1)
	        Bnorm[p]=Bnorm[p]+phi[p*Q+q]*phi[p*Q+q];
	    }
    Bnorm[p]=sqrt(Bnorm[p]);
}
//////////////////////////////
double Dist(int P, int Q, double *phi_last, double * phi)
{
	int p,q;
	double temp,result;	
      result=0;
	for(p=0;p<P;p++)
	 for(q=0;q<Q;q++)
	  {
		  temp=phi_last[p*Q+q]-phi[p*Q+q];
		  temp=fabs(temp);
		  if(temp>result)
		    result=temp;
      }
    return(result);
}
