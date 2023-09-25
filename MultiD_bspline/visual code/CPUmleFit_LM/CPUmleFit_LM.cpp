//Copyright (c) 2023 Li Lab, Southern University of Science and Technology.
//author: Yiming Li
//email: yiming.li@embl.de
//date: 2023.09.22


//Terms of Use
//
//This file is part of GPUmleFit_LM_multiD.
//
//GPUmleFit_LM_multiD Fitter is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
//GPUmleFit_LM_multiD Fitter is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License along with GPUmleFit_LM Fitter. If not, see <http://www.gnu.org/licenses/>.
//
//Additional permission under GNU GPL version 3 section 7



/*!
 * \file CPUmleFit_LM_multiD.cpp
 * \brief This contains the definitions for all the fitting mode.  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include "definitions.h"
#include "MatInvLib.h"
#include <math.h>
#include "CPUsplineLib.h"
#include "CPUgaussLib.h"
#include "CPUmleFit_LM.h"

//*******************************************************************************************
//theta is: {x,y,N,bg}
 void kernel_MLEFit_LM_EMCCD(const int subregion, const float *d_data,const float PSFSigma, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits){
		/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */

		const int NV=NV_P;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		//int tx = threadIdx.x;
		//int bx = blockIdx.x;
		//int BlockSize = blockDim.x;
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
	    float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma*PSFSigma);
		newTheta[3] = max(newTheta[3],0.01);

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);

		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D(ii,jj,PSFSigma,newTheta,newDudt,&model);
			data=s_data[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;

				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeGauss2D(ii,jj,PSFSigma,newTheta,newDudt,&model);
						data=s_data[sz*jj+ii];			

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		//output iteration
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D(ii,jj,PSFSigma,newTheta,newDudt,&model);
			data=s_data[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//*********************************************************************************************************************************************

 void kernel_MLEFit_LM_Sigma_EMCCD(const int subregion, const float *d_data,const float PSFSigma, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits){
		/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */

		const int NV=NV_PS;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		/*int tx = threadIdx.x;
		int bx = blockIdx.x;
		int BlockSize = blockDim.x;*/
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
		float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20,0.5};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma*PSFSigma);
		newTheta[3] = max(newTheta[3],0.01);
		newTheta[4]=PSFSigma;

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);;


		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D_sigma(ii,jj,newTheta,newDudt,&model);
			data=s_data[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					newTheta[4] = max(newTheta[4],0.0);
					newTheta[4] = min(newTheta[4],sz/2.0f);


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeGauss2D_sigma(ii,jj,newTheta,newDudt,&model);
						data=s_data[sz*jj+ii];			

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		//output iteration
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			
			kernel_DerivativeGauss2D_sigma(ii,jj,newTheta,newDudt,&model);
			data=s_data[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//*********************************************************************************************************************************************

 void kernel_MLEFit_LM_z_EMCCD(const int subregion,const float *d_data, const float PSFSigma_x, const float Ax, const float Ay, const float Bx, 
	const float By, const float gamma, const float d, const float PSFSigma_y, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits){
			/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma_x sigma of the point spread function on the x axis
	 * \param Ax ???
	 * \param Ay ???
	 * \param Bx ???
	 * \param By ???
	 * \param gamma ???
	 * \param d ???
	 * \param PSFSigma_y sigma of the point spread function on the y axis
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */

		const int NV=NV_PZ;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		//int tx = threadIdx.x;
		//int bx = blockIdx.x;
		//int BlockSize = blockDim.x;
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;
		float PSFy, PSFx;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
		float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20,2};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
//		float temp;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma_x, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma_x*PSFSigma_y*sqrt(2.0f));
		newTheta[3] = max(newTheta[3],0.01);
		newTheta[4]=0;

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);

		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			 kernel_DerivativeIntGauss2Dz(ii, jj, newTheta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, newDudt, NULL,&model);
			data=s_data[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}

					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeIntGauss2Dz(ii, jj, newTheta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, newDudt, NULL,&model);
						data=s_data[sz*jj+ii];			

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		//output iteration
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
		    kernel_DerivativeIntGauss2Dz(ii, jj, newTheta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, newDudt, NULL,&model);
			data=s_data[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//*********************************************************************************************************************************************

 void kernel_MLEFit_LM_sigmaxy_EMCCD(const int subregion,const float *d_data, const float PSFSigma, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits){
			/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */

		const int NV=NV_PS2;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		//int tx = threadIdx.x;
		//int bx = blockIdx.x;
		//int BlockSize = blockDim.x;
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
		float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20,0.5,0.5};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		//float temp;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma*PSFSigma);
		newTheta[3] = max(newTheta[3],0.01);
		newTheta[4]=PSFSigma;
		newTheta[5]=PSFSigma;

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);

		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D_sigmaxy( ii,  jj, newTheta, newDudt, &model);
			data=s_data[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					newTheta[4] = max(newTheta[4],PSFSigma/10.0f);
					newTheta[5] = max(newTheta[5],PSFSigma/10.0f);

					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeGauss2D_sigmaxy( ii,  jj, newTheta, newDudt, &model);
						data=s_data[sz*jj+ii];			

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			//need to check why don't use newTheta[4] instead of PSFSigma!!!
			kernel_DerivativeGauss2D_sigmaxy( ii,  jj, newTheta, newDudt, &model);
			data=s_data[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//******************************************************************************************************

void kernel_splineMLEFit_z_EMCCD(const int subregion,const float *d_data,const float *d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,float initZ, const int Nfits){
		/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param d_coeff array of spline coefficients of the PSF model
	 * \param spline_xsize,spline_ysize,spline_zsize, x,y,z size of spline coefficients
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param initZ intial z position used for fitting
	 * \param Nfits number of subregions to fit
	 */
	
	
   const int NV=NV_PSP;
    float M[NV*NV],Diag[NV], Minv[NV*NV];
    //int tx = threadIdx.x;
    //int bx = blockIdx.x;
    //int BlockSize = blockDim.x;
    int ii, jj, kk, ll, l, m, i;
	int xstart, ystart, zstart;

	const float *s_coeff;
	s_coeff = d_coeff;

    float model, data;
    float Div;
    //float dudt[NV_PS];
    float newTheta[NV],oldTheta[NV];
	float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
	float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13};
	float maxJump[NV]={1.0,1.0,100,20,2};
	float newDudt[NV] ={0};

	float newErr = 1e12, oldErr = 1e13;

	float off;
	float jacobian[NV]={0};
	float hessian[NV*NV]={0};
	float t1,t2;

	float Nmax;
	float xc,yc,zc;
	float delta_f[64]={0}, delta_dxf[64]={0}, delta_dyf[64]={0}, delta_dzf[64]={0};
	int errFlag=0;
	float L[NV*NV] = {0}, U[NV*NV] = {0};

    
    //Prevent read/write past end of array
    if (subregion>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

    //copy in data
      const float *s_data = d_data+(sz*sz*subregion);
    
    //initial values
    kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
    kernel_GaussFMaxMin2D(sz, 1.5, s_data, &Nmax, &newTheta[3]);
	//central pixel of spline model
	newTheta[3] = max(newTheta[3],0.01);
	newTheta[2]= (Nmax-newTheta[3])/d_coeff[(int)(spline_zsize/2)*(spline_xsize*spline_ysize)+(int)(spline_ysize/2)*spline_xsize+(int)(spline_xsize/2)]*4;

    newTheta[4]=initZ;

	maxJump[2]=max(newTheta[2],maxJump[2]);

	maxJump[3]=max(newTheta[3],maxJump[3]);

	maxJump[4]= max(spline_zsize/3.0f,maxJump[4]);


	for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

	//updateFitValues
	xc = -1.0*((newTheta[0]-float(sz)/2)+0.5);
	yc = -1.0*((newTheta[1]-float(sz)/2)+0.5);

	off = floor((float(spline_xsize)+1.0-float(sz))/2);

	xstart = floor(xc);
	xc = xc-xstart;

	ystart = floor(yc);
	yc = yc-ystart;

	//zstart = floor(newTheta[4]);
	zstart = floor(newTheta[4]);
	zc = newTheta[4] -zstart;

	newErr = 0;
	memset(jacobian,0,NV*sizeof(float));
	memset(hessian,0,NV*NV*sizeof(float));
	kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf);

	for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
		kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		data=s_data[sz*jj+ii];

		if (data>0)
			newErr = newErr + 2*((model-data)-data*log(model/data));
		else
		{
			newErr = newErr + 2*model;
			data = 0;
		}

		t1 = 1-data/model;
		for (l=0;l<NV;l++){
			jacobian[l]+=t1*newDudt[l];
		}

		t2 = data/pow(model,2);
		for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
			hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
			hessian[m*NV+l] = hessian[l*NV+m];
		}
	}

	for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata
					
					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}
				

				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5f;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2.0f-sz/4.0f);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2.0f+sz/4.0f);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2.0f-sz/4.0f);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2.0f+sz/4.0f);
					newTheta[2] = max(newTheta[2],1.0f);
					newTheta[3] = max(newTheta[3],0.01f);
					newTheta[4] = max(newTheta[4],0.0f);
					newTheta[4] = min(newTheta[4],float(spline_zsize));

					//updateFitValues
					xc = -1.0*((newTheta[0]-float(sz)/2)+0.5f);
					yc = -1.0*((newTheta[1]-float(sz)/2)+0.5f);

					xstart = floor(xc);
					xc = xc-xstart;

					ystart = floor(yc);
					yc = yc-ystart;

					//zstart = floor(newTheta[4]);
					zstart = floor(newTheta[4]);
					zc = newTheta[4] -zstart;


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf);
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
						data=s_data[sz*jj+ii];

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}	
	}
	//output iteration time
	d_Parameters[Nfits*NV+subregion]=kk;
    
    // Calculating the CRLB and LogLikelihood
	Div=0.0;

	xc = -1.0*((newTheta[0]-float(sz)/2)+0.5);
	yc = -1.0*((newTheta[1]-float(sz)/2)+0.5);

	xstart = floor(xc);
	xc = xc-xstart;

	ystart = floor(yc);
	yc = yc-ystart;

	//zstart = floor(newTheta[4]);
	zstart = floor(newTheta[4]);
	zc = newTheta[4] -zstart;

	kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf);

    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
		kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		data=s_data[sz*jj+ii];
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
  
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
   for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
   d_LogLikelihood[subregion] = Div;
    return;
}


//*********************************************************************************************************************************************
 void kernel_MLEFit_LM_sCMOS(const int subregion, const float *d_data,const float PSFSigma, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits,const float *d_varim){
		/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 * \d_varim variance map of scmos
	 */

		const int NV=NV_P;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		//int tx = threadIdx.x;
		//int bx = blockIdx.x;
		//int BlockSize = blockDim.x;
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
	    float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);
		const float *s_varim = d_varim+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma*PSFSigma);
		newTheta[3] = max(newTheta[3],0.01);

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);

		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D(ii,jj,PSFSigma,newTheta,newDudt,&model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;

				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*newLambda;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeGauss2D(ii,jj,PSFSigma,newTheta,newDudt,&model);
						model +=s_varim[sz*jj+ii];
						data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];		

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		//output iteration
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D(ii,jj,PSFSigma,newTheta,newDudt,&model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//*********************************************************************************************************************************************

 void kernel_MLEFit_LM_Sigma_sCMOS(const int subregion, const float *d_data,const float PSFSigma, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits,const float *d_varim){
		/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 * \d_varim variance map of scmos
	 */
		const int NV=NV_PS;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		/*int tx = threadIdx.x;
		int bx = blockIdx.x;
		int BlockSize = blockDim.x;*/
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
		float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20,0.5};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);
		const float *s_varim = d_varim+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma*PSFSigma);
		newTheta[3] = max(newTheta[3],0.01);
		newTheta[4]=PSFSigma;

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);


		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D_sigma(ii,jj,newTheta,newDudt,&model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;

				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					newTheta[4] = max(newTheta[4],0.0);
					newTheta[4] = min(newTheta[4],sz/2.0f);


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeGauss2D_sigma(ii,jj,newTheta,newDudt,&model);
						model +=s_varim[sz*jj+ii];
						data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];		

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D_sigma(ii,jj,newTheta,newDudt,&model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//*********************************************************************************************************************************************

 void kernel_MLEFit_LM_z_sCMOS(const int subregion,const float *d_data, const float PSFSigma_x, const float Ax, const float Ay, const float Bx, 
	const float By, const float gamma, const float d, const float PSFSigma_y, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits,const float *d_varim){
			/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma_x sigma of the point spread function on the x axis
	 * \param Ax ???
	 * \param Ay ???
	 * \param Bx ???
	 * \param By ???
	 * \param gamma ???
	 * \param d ???
	 * \param PSFSigma_y sigma of the point spread function on the y axis
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 * \d_varim variance map of scmos
	 */

		const int NV=NV_PZ;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		//int tx = threadIdx.x;
		//int bx = blockIdx.x;
		//int BlockSize = blockDim.x;
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;
		float PSFy, PSFx;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
		float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20,2};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);
		const float *s_varim = d_varim+(sz*sz*subregion);
		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma_x, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma_x*PSFSigma_y*sqrt(2.0f));
		newTheta[3] = max(newTheta[3],0.01);
		newTheta[4]=0;

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);

		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			 kernel_DerivativeIntGauss2Dz(ii, jj, newTheta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, newDudt, NULL,&model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}

				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					//newTheta[4] = max(newTheta[4],0.0);
					//newTheta[4] = min(newTheta[4],sz/2.0f);


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeIntGauss2Dz(ii, jj, newTheta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, newDudt, NULL,&model);
						model +=s_varim[sz*jj+ii];
						data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];		

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
		    kernel_DerivativeIntGauss2Dz(ii, jj, newTheta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, newDudt, NULL,&model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//*********************************************************************************************************************************************

 void kernel_MLEFit_LM_sigmaxy_sCMOS(const int subregion,const float *d_data, const float PSFSigma, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits,const float *d_varim){
			/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	  * \d_varim variance map of scmos
	 */

		const int NV=NV_PS2;
		float M[NV*NV],Diag[NV], Minv[NV*NV];
		//int tx = threadIdx.x;
		//int bx = blockIdx.x;
		//int BlockSize = blockDim.x;
		int ii, jj, kk, ll, l, m, i;


		float model, data;
		float Div;

		float newTheta[NV],oldTheta[NV];
		float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
		float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13, 1e13};
		float maxJump[NV]={1.0,1.0,100,20,0.5,0.5};
		float newDudt[NV] ={0};

		float newErr = 1e12, oldErr = 1e13;

		float jacobian[NV]={0};
		float hessian[NV*NV]={0};
		float t1,t2;

		float Nmax;
		int errFlag=0;
		float L[NV*NV] = {0}, U[NV*NV] = {0};


		//Prevent read/write past end of array
		if (subregion>=Nfits) return;

		for (ii=0;ii<NV*NV;ii++)M[ii]=0;
		for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

		//copy in data
		const float *s_data = d_data+(sz*sz*subregion);
		const float *s_varim = d_varim+(sz*sz*subregion);

		//initial values
		kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &newTheta[3]);
		newTheta[2]=max(0.0, (Nmax-newTheta[3])*2*PI*PSFSigma*PSFSigma);
		newTheta[3] = max(newTheta[3],0.01);
		newTheta[4]=PSFSigma;
		newTheta[5]=PSFSigma;

		maxJump[2]=max(newTheta[2],maxJump[2]);

		maxJump[3]=max(newTheta[3],maxJump[3]);

		for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

		//updateFitValues
		newErr = 0;
		memset(jacobian,0,NV*sizeof(float));
		memset(hessian,0,NV*NV*sizeof(float));
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D_sigmaxy( ii,  jj, newTheta, newDudt, &model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			if (data>0)
				newErr = newErr + 2*((model-data)-data*log(model/data));
			else
			{
				newErr = newErr + 2*model;
				data = 0;
			}

			t1 = 1-data/model;
			for (l=0;l<NV;l++){
				jacobian[l]+=t1*newDudt[l];
			}

			t2 = data/pow(model,2);
			for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
				hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
				hessian[m*NV+l] = hessian[l*NV+m];
			}
		}

		for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata

					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}


				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					newTheta[4] = max(newTheta[4],PSFSigma/10.0f);
					newTheta[5] = max(newTheta[5],PSFSigma/10.0f);

					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						//calculating derivatives
						kernel_DerivativeGauss2D_sigmaxy( ii,  jj, newTheta, newDudt, &model);
						model +=s_varim[sz*jj+ii];
						data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];			

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
			}
		}
		//output iteration
		d_Parameters[Nfits*NV+subregion]=kk;
		// Calculating the CRLB and LogLikelihood
		Div=0.0f;
		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			kernel_DerivativeGauss2D_sigmaxy( ii,  jj, newTheta, newDudt, &model);
			model +=s_varim[sz*jj+ii];
			data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];

			//Building the Fisher Information Matrix
			for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
				M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
				M[ll*NV+kk]=M[kk*NV+ll];
			}

			//LogLikelyhood
			if (model>0)
				if (data>0)Div+=data*log(model)-model-data*log(data)+data;
				else
					Div+=-model;
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
		kernel_MatInvN(M, Minv, Diag, NV);
		//write to global arrays
		for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
		for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
		d_LogLikelihood[subregion] = Div;

		return;
}

//******************************************************************************************************

void kernel_splineMLEFit_z_sCMOS(const int subregion,const float *d_data,const float *d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,float initZ, const int Nfits,const float *d_varim){
		/*! 
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param d_coeff array of spline coefficients of the PSF model
	 * \param spline_xsize,spline_ysize,spline_zsize, x,y,z size of spline coefficients
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param initZ intial z position used for fitting
	 * \param Nfits number of subregions to fit
	 * \param d_varim variance map of sCMOS
	 */
	
   const int NV=NV_PSP;
    float M[NV*NV],Diag[NV], Minv[NV*NV];
    //int tx = threadIdx.x;
    //int bx = blockIdx.x;
    //int BlockSize = blockDim.x;
    int ii, jj, kk, ll, l, m, i;
	int xstart, ystart, zstart;

	const float *s_coeff;
	s_coeff = d_coeff;

    float model, data;
    float Div;
    float newTheta[NV],oldTheta[NV];
	float newLambda = INIT_LAMBDA, oldLambda = INIT_LAMBDA, mu;
	float newUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13},oldUpdate[NV] = {1e13, 1e13, 1e13, 1e13, 1e13};
	float maxJump[NV]={1.0,1.0,100,20,2};
	float newDudt[NV] ={0};

	float newErr = 1e12, oldErr = 1e13;

	float off;
	float jacobian[NV]={0};
	float hessian[NV*NV]={0};
	float t1,t2;

	float Nmax;
	float xc,yc,zc;
	float delta_f[64]={0}, delta_dxf[64]={0}, delta_dyf[64]={0}, delta_dzf[64]={0};
	int errFlag=0;
	float L[NV*NV] = {0}, U[NV*NV] = {0};

    
    //Prevent read/write past end of array
    if (subregion>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

    //copy in data
      const float *s_data = d_data+(sz*sz*subregion);
	  const float *s_varim = d_varim+(sz*sz*subregion);
    
    //initial values
    kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
    kernel_GaussFMaxMin2D(sz, 1.5, s_data, &Nmax, &newTheta[3]);

	//central pixel of spline model
	newTheta[3] = max(newTheta[3],0.01);
	newTheta[2]= (Nmax-newTheta[3])/d_coeff[(int)(spline_zsize/2)*(spline_xsize*spline_ysize)+(int)(spline_ysize/2)*spline_xsize+(int)(spline_xsize/2)]*4;

    newTheta[4]=initZ;

	maxJump[2]=max(newTheta[2],maxJump[2]);

	maxJump[3]=max(newTheta[3],maxJump[3]);

	maxJump[4]= max(spline_zsize/3.0f,maxJump[4]);


	for (ii=0;ii<NV;ii++)oldTheta[ii]=newTheta[ii];

	//updateFitValues
	xc = -1.0*((newTheta[0]-float(sz)/2)+0.5);
	yc = -1.0*((newTheta[1]-float(sz)/2)+0.5);

	off = floor((float(spline_xsize)+1.0-float(sz))/2);

	xstart = floor(xc);
	xc = xc-xstart;

	ystart = floor(yc);
	yc = yc-ystart;

	//zstart = floor(newTheta[4]);
	zstart = floor(newTheta[4]);
	zc = newTheta[4] -zstart;

	newErr = 0;
	memset(jacobian,0,NV*sizeof(float));
	memset(hessian,0,NV*NV*sizeof(float));
	kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf);

	for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
		kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		model +=s_varim[sz*jj+ii];
		data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];	

		if (data>0)
			newErr = newErr + 2*((model-data)-data*log(model/data));
		else
		{
			newErr = newErr + 2*model;
			data = 0;
		}

		t1 = 1-data/model;
		for (l=0;l<NV;l++){
			jacobian[l]+=t1*newDudt[l];
		}

		t2 = data/pow(model,2);
		for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
			hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
			hessian[m*NV+l] = hessian[l*NV+m];
		}
	}

	for (kk=0;kk<iterations;kk++) {//main iterative loop

			if(fabs((newErr-oldErr)/newErr)<TOLERANCE){
				//CONVERGED;
				break;
			}
			else{
				if(newErr>ACCEPTANCE*oldErr){
					//copy Fitdata
					for (i=0;i<NV;i++){
						newTheta[i]=oldTheta[i];
						newUpdate[i]=oldUpdate[i];
					}
					newLambda = oldLambda;
					newErr = oldErr;
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}
				else if(newErr<oldErr&&errFlag==0){
					newLambda = SCALE_DOWN*newLambda;
				    mu = 1+newLambda;
				}
				

				for (i=0;i<NV;i++){
					hessian[i*NV+i]=hessian[i*NV+i]*mu;
				}
				memset(L,0,NV*sizeof(float));
				memset(U,0,NV*sizeof(float));
				errFlag = kernel_cholesky(hessian,NV,L,U);
				if (errFlag ==0){
					for (i=0;i<NV;i++){
						oldTheta[i]=newTheta[i];
						oldUpdate[i] = newUpdate[i];
					}
					oldLambda = newLambda;
					oldErr=newErr;

					kernel_luEvaluate(L,U,jacobian,NV,newUpdate);	
					
					//updateFitParameters
					for (ll=0;ll<NV;ll++){
						if (newUpdate[ll]/oldUpdate[ll]< -0.5f){
							maxJump[ll] = maxJump[ll]*0.5;
						}
					    newUpdate[ll] = newUpdate[ll]/(1+fabs(newUpdate[ll]/maxJump[ll]));
						newTheta[ll] = newTheta[ll]-newUpdate[ll];
					}
					//restrict range
					newTheta[0] = max(newTheta[0],(float(sz)-1)/2-sz/4.0);
					newTheta[0] = min(newTheta[0],(float(sz)-1)/2+sz/4.0);
					newTheta[1] = max(newTheta[1],(float(sz)-1)/2-sz/4.0);
					newTheta[1] = min(newTheta[1],(float(sz)-1)/2+sz/4.0);
					newTheta[2] = max(newTheta[2],1.0);
					newTheta[3] = max(newTheta[3],0.01);
					newTheta[4] = max(newTheta[4],0.0);
					newTheta[4] = min(newTheta[4],float(spline_zsize));

					//updateFitValues
					xc = -1.0*((newTheta[0]-float(sz)/2)+0.5);
					yc = -1.0*((newTheta[1]-float(sz)/2)+0.5);

					xstart = floor(xc);
					xc = xc-xstart;

					ystart = floor(yc);
					yc = yc-ystart;

					zstart = floor(newTheta[4]);
					zc = newTheta[4] -zstart;


					newErr = 0;
					memset(jacobian,0,NV*sizeof(float));
					memset(hessian,0,NV*NV*sizeof(float));
					kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf);
					for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
						kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
						model +=s_varim[sz*jj+ii];
						data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];	

						if (data>0)
							newErr = newErr + 2*((model-data)-data*log(model/data));
						else
						{
							newErr = newErr + 2*model;
							data = 0;
						}

						t1 = 1-data/model;
						for (l=0;l<NV;l++){
							jacobian[l]+=t1*newDudt[l];
						}

						t2 = data/pow(model,2);
						for (l=0;l<NV;l++) for(m=l;m<NV;m++) {
							hessian[l*NV+m] +=t2*newDudt[l]*newDudt[m];
							hessian[m*NV+l] = hessian[l*NV+m];
						}
					}
				}
				else
				{
					mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3f);         
					newLambda = SCALE_UP*newLambda;
				}

			}


		
	}
	//output iteration time
	d_Parameters[Nfits*NV+subregion]=kk;
    
    // Calculating the CRLB and LogLikelihood
	Div=0.0;

	xc = -1.0*((newTheta[0]-float(sz)/2)+0.5);
	yc = -1.0*((newTheta[1]-float(sz)/2)+0.5);

	//off = (float(spline_xsize)+1.0-2*float(sz))/2;

	xstart = floor(xc);
	xc = xc-xstart;

	ystart = floor(yc);
	yc = yc-ystart;

	//zstart = floor(newTheta[4]);
	zstart = floor(newTheta[4]);
	zc = newTheta[4] -zstart;

	kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf);

    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
		kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		model +=s_varim[sz*jj+ii];
		data=s_data[sz*jj+ii]+s_varim[sz*jj+ii];	
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= newDudt[ll]*newDudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
  
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+subregion]=newTheta[kk];
   for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+subregion]=Diag[kk];
   d_LogLikelihood[subregion] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;
    
    
    return;
}


void kernel_bsplineMLEFit_z_EMCCD(const int subregion, const float* d_data, const float* d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int sz, const int iterations,
	float* d_Parameters, float* d_CRLBs, float* d_LogLikelihood, const int Nfits, float* deg) {

	//__shared__ float s_coeff[17*17*25];

	const int NV = NV_PSP;
	float M[NV * NV], Diag[NV], Minv[NV * NV];
	//int tx = threadIdx.x;
	//int bx = blockIdx.x;
	//int BlockSize = blockDim.x;
	int ii, jj, kk, ll, l, m, i;
	int xstart, ystart, zstart, xi, yi;

	const float* s_coeff;
	s_coeff = d_coeff;

	float model, data;
	float Div;
	//float dudt[NV_PS];
	float newTheta[NV], oldTheta[NV];
	float newLambda = 1.0, oldLambda = 1.0;
	float newSign[NV] = { 0 }, oldSign[NV] = { 0 };
	float newUpdate[NV] = { 0 }, oldUpdate[NV] = { 0 };
	float newClamp[NV] = { 2.0,2.0,10000,20,20 }, oldClamp[NV] = { 2.0,2.0,10000,20,20 };
	float newDudt[NV] = { 0 };

	float newErr = 1e12, oldErr = 1e13;

	float off;
	float jacobian[NV] = { 0 };
	float hessian[NV * NV] = { 0 };
	float t1, t2;

	float Nmax;
	float xc, yc, zc;
	float xs, ys, zs;
	float temp;
	float delta_f[64] = { 0 }, delta_dxf[64] = { 0 }, delta_dyf[64] = { 0 }, delta_dzf[64] = { 0 };
	int info;
	float L[NV * NV] = { 0 }, U[NV * NV] = { 0 };


	//Prevent read/write past end of array
	if (subregion >= Nfits) return;

	//for (ii=0;ii<spline_xsize*spline_ysize*spline_zsize;ii++)
	//       s_coeff[ii]=d_coeff[ii];

	 //if (threadIdx.x < 256) {
  //      for(int ii = threadIdx.x; ii  <spline_xsize*spline_ysize*spline_zsize; i += 256) {
  //          s_coeff[ii]=d_coeff[ii];
  //      }
  //  }
	 //__syncthreads();
	//__syncthread();


	for (ii = 0; ii < NV * NV; ii++)M[ii] = 0;
	for (ii = 0; ii < NV * NV; ii++)Minv[ii] = 0;

	//copy in data
	const float* s_data = d_data + (sz * sz * subregion);

	//const float *s_varim = d_varim+(sz*sz*bx*BlockSize+sz*sz*tx);
  //const float *s_gainim = d_gainim+(sz*sz*bx*BlockSize+sz*sz*tx);

  //initial values
	kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
	kernel_GaussFMaxMin2D(sz, 1.5, s_data, &Nmax, &newTheta[3]);
	newTheta[2] = max(0.0, (Nmax - newTheta[3]) * 2 * PI * 1.5 * 1.5);
	newTheta[3] = max(newTheta[3], 0.01);
	newTheta[4] = float(spline_zsize) / 2.0f;
	//newClamp[2] = max(newTheta[2], newClamp[2]);
	//oldClamp[2] = newClamp[2];

	//newClamp[3] = max(newTheta[3], newClamp[3]);
	//oldClamp[3] = newClamp[3];

	//newClamp[4] = max(spline_zsize / 3.0f, newClamp[4]);
	//oldClamp[4] = newClamp[4];

	for (ii = 0; ii < NV; ii++)oldTheta[ii] = newTheta[ii];

	//updateFitValues3D

	//xc = -2.0*((newTheta[0]-float(sz)/2)+0.5);
	//yc = -2.0*((newTheta[1]-float(sz)/2)+0.5);

	//off = (float(spline_xsize)+1.0-2*float(sz))/2;

	//xstart = floor(xc);
	//xc = xc-xstart;

	//ystart = floor(yc);
	//yc = yc-ystart;

	////zstart = floor(newTheta[4]);
	//zstart = floor(newTheta[4]);
	//zc = newTheta[4] -zstart;

	//flipped in the bSpline
	xc = (sz + 1) / 2 - newTheta[0];
	yc = (sz + 1) / 2 - newTheta[1];
	zc = newTheta[4];
	//off = (spline_xsize - 4 - sz) / 2.0f;
	off = (spline_xsize - sz) / 2.0f;//new
	xs = xc - floor(xc);
	ys = yc - floor(yc);
	zs = zc - floor(zc);

	newErr = 0.0f;
	memset(jacobian, 0, NV * sizeof(float));
	memset(hessian, 0, NV * NV * sizeof(float));
	kernel_computeDelta3D_bSpline(xs, ys, zs, delta_f, delta_dxf, delta_dyf, delta_dzf);

	for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
		//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		kernel_Derivative_bSpline1(xc + jj + off, yc + ii + off, zc, spline_xsize, spline_ysize, spline_zsize, delta_f, delta_dxf, delta_dyf, delta_dzf, s_coeff, newTheta, newDudt, &model);

		data = s_data[sz * jj + ii];

		if (data > 0)
			newErr = newErr + 2 * ((model - data) - data * log(model / data));
		else
		{
			newErr = newErr + 2 * model;
			data = 0;
		}

		t1 = 1 - data / model;
		for (l = 0; l < NV; l++) {
			jacobian[l] += t1 * newDudt[l];
		}

		t2 = data / pow(model, 2);
		for (l = 0; l < NV; l++) for (m = l; m < NV; m++) {
			hessian[l * NV + m] += t2 * newDudt[l] * newDudt[m];
			hessian[m * NV + l] = hessian[l * NV + m];
		}
	}
	//addPeak

	//copyFitData

	for (kk = 0; kk < iterations; kk++) {//main iterative loop

		if (fabs((newErr - oldErr) / newErr) < TOLERANCE) {
			//newStatus = CONVERGED;
			/*deg[90]=1;*/
			break;
		}
		else {
			if (newErr > 1.5 * oldErr) {
				//copy Fitdata

				for (i = 0; i < NV; i++) {
					newSign[i] = oldSign[i];
					newClamp[i] = oldClamp[i];
					newTheta[i] = oldTheta[i];
				}
				newLambda = oldLambda;
				newErr = oldErr;

				newLambda = 10 * newLambda;
			}
			else if (newErr < oldErr) {

				if (newLambda > 1) {
					newLambda = newLambda * 0.8;
				}
				else if (newLambda < 1) {
					newLambda = 1;
				}
			}


			for (i = 0; i < NV; i++) {
				hessian[i * NV + i] = hessian[i * NV + i] * newLambda;
			}
			memset(L, 0, NV * sizeof(float));
			memset(U, 0, NV * sizeof(float));
			info = kernel_cholesky(hessian, NV, L, U);
			if (info == 0) {
				kernel_luEvaluate(L, U, jacobian, NV, newUpdate);
				//copyFitData
				for (i = 0; i < NV; i++) {
					oldSign[i] = newSign[i];
					oldClamp[i] = newClamp[i];

					oldTheta[i] = newTheta[i];
				}
				oldLambda = newLambda;
				oldErr = newErr;


				//updatePeakParameters
				for (ll = 0; ll < NV; ll++) {
					if (newSign[ll] != 0) {
						if (newSign[ll] == 1 && newUpdate[ll] < 0) {
							newClamp[ll] = newClamp[ll] * 0.5;
						}
						else if (newSign[ll] == -1 && newUpdate[ll] > 0) {
							newClamp[ll] = newClamp[ll] * 0.5;
						}
					}

					if (newUpdate[ll] > 0) {
						newSign[ll] = 1;
					}
					else {
						newSign[ll] = -1;
					}

					newTheta[ll] = newTheta[ll] - newUpdate[ll] / (1 + fabs(newUpdate[ll] / newClamp[ll]));
				}

				//newTheta[0] = max(newTheta[0], (float(sz) - 1) / 2 - sz / 4.0);
				//newTheta[0] = min(newTheta[0], (float(sz) - 1) / 2 + sz / 4.0);
				//newTheta[1] = max(newTheta[1], (float(sz) - 1) / 2 - sz / 4.0);
				//newTheta[1] = min(newTheta[1], (float(sz) - 1) / 2 + sz / 4.0);
				newTheta[2] = max(newTheta[2], 1.0);
				newTheta[3] = max(newTheta[3], 0.01);
				newTheta[4] = max(newTheta[4], 0.0);
				newTheta[4] = min(newTheta[4], float(spline_zsize - 4));

				//updateFitValues3D
				//xc = -2.0*((newTheta[0]-float(sz)/2)+0.5);
				//yc = -2.0*((newTheta[1]-float(sz)/2)+0.5);

				//off = (float(spline_xsize)+1.0-2*float(sz))/2;

				//xstart = floor(xc);
				//xc = xc-xstart;

				//ystart = floor(yc);
				//yc = yc-ystart;

				////zstart = floor(newTheta[4]);
				//zstart = floor(newTheta[4]);
				//zc = newTheta[4] -zstart;
				//flipped in the bSpline
				xc = (sz + 1) / 2 - newTheta[0];
				yc = (sz + 1) / 2 - newTheta[1];
				zc = newTheta[4];
				//off = float(spline_xsize-4-sz)/2.0f;
				xs = xc - floor(xc);
				ys = yc - floor(yc);
				zs = zc - floor(zc);


				newErr = 0;
				memset(jacobian, 0, NV * sizeof(float));
				memset(hessian, 0, NV * NV * sizeof(float));
				kernel_computeDelta3D_bSpline(xs, ys, zs, delta_f, delta_dxf, delta_dyf, delta_dzf);

				for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
					//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
					kernel_Derivative_bSpline1(xc + jj + off, yc + ii + off, zc, spline_xsize, spline_ysize, spline_zsize, delta_f, delta_dxf, delta_dyf, delta_dzf, s_coeff, newTheta, newDudt, &model);

					//temp = kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,s_coeff);
					//model = newTheta[3]+newTheta[2]*temp;
					data = s_data[sz * jj + ii];
					//calculating derivatives

					//newDudt[0] = -1*newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dxf,s_coeff);
					//newDudt[1] = -1*newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dyf,s_coeff);
					//newDudt[4] = newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dzf,s_coeff);
					//newDudt[2] = temp;
					//newDudt[3] = 1;

					if (data > 0)
						newErr = newErr + 2 * ((model - data) - data * log(model / data));
					else
					{
						newErr = newErr + 2 * model;
						data = 0;
					}

					t1 = 1 - data / model;
					for (l = 0; l < NV; l++) {
						jacobian[l] += t1 * newDudt[l];
					}

					t2 = data / pow(model, 2);
					for (l = 0; l < NV; l++) for (m = l; m < NV; m++) {
						hessian[l * NV + m] += t2 * newDudt[l] * newDudt[m];
						hessian[m * NV + l] = hessian[l * NV + m];
					}
				}
			}
			else
			{
				newLambda = 10 * newLambda;
			}

			//copyFitdata	

		}



	}
	d_Parameters[Nfits * NV + subregion] = kk;
	// Calculating the CRLB and LogLikelihood
	Div = 0.0;

	xc = (sz + 1) / 2 - newTheta[0];
	yc = (sz + 1) / 2 - newTheta[1];
	zc = newTheta[4];
	//off = float(spline_xsize-4-sz)/2.0f;
	xs = xc - floor(xc);
	ys = yc - floor(yc);
	zs = zc - floor(zc);

	kernel_computeDelta3D_bSpline(xs, ys, zs, delta_f, delta_dxf, delta_dyf, delta_dzf);




	for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
		kernel_Derivative_bSpline1(xc + jj + off, yc + ii + off, zc, spline_xsize, spline_ysize, spline_zsize, delta_f, delta_dxf, delta_dyf, delta_dzf, s_coeff, newTheta, newDudt, &model);
		//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		data = s_data[sz * jj + ii];

		//Building the Fisher Information Matrix
		for (kk = 0; kk < NV; kk++)for (ll = kk; ll < NV; ll++) {
			M[kk * NV + ll] += newDudt[ll] * newDudt[kk] / model;
			M[ll * NV + kk] = M[kk * NV + ll];
		}

		//LogLikelyhood
		if (model > 0)
			if (data > 0)Div += data * log(model) - model - data * log(data) + data;
			else
				Div += -model;
	}

	// Matrix inverse (CRLB=F^-1) and output assigments
	kernel_MatInvN(M, Minv, Diag, NV);


	//write to global arrays
	for (kk = 0; kk < NV; kk++) d_Parameters[Nfits * kk + subregion] = newTheta[kk];
	for (kk = 0; kk < NV; kk++) d_CRLBs[Nfits * kk + subregion] = Diag[kk];
	d_LogLikelihood[subregion] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;


	return;
}

//****************/4D
void kernel_bsplineMLEFit_z_EMCCD_4D(const int subregion, const float* d_data, const float* d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int spline_wsize, const int sz, const int iterations, const float* initP4,
	float* d_Parameters, float* d_CRLBs, float* d_LogLikelihood, const int Nfits, float* deg) {

	//__shared__ float s_coeff[17*17*25];

	const int NV = NV_4D;
	float M[NV * NV], Diag[NV], Minv[NV * NV];
	//int tx = threadIdx.x;
	//int bx = blockIdx.x;
	//int BlockSize = blockDim.x;
	int ii, jj, kk, ll, l, m, i;
	int xstart, ystart, zstart, wstart, xi, yi, turn = 1;

	const float* s_coeff;
	s_coeff = d_coeff;

	float model, data;
	float Div;
	//float dudt[NV_PS];
	float newTheta[NV], oldTheta[NV];
	float newLambda = 1.0, oldLambda = 1.0;
	float newSign[NV] = { 0 }, oldSign[NV] = { 0 };
	float newUpdate[NV] = { 0 }, oldUpdate[NV] = { 0 };
	float newClamp[NV] = { 1.0,1.0,10000,20,2,0.00001 }, oldClamp[NV] = { 1.0,1.0,10000,20,2,0.00001 };
	float newDudt[NV] = { 0 };
	float  fade[NV] = { 0.5,0.5,0.4,0.5,0.7,0.8 };

	float newErr = 1e12, oldErr = 1e13;

	float off;
	float jacobian[NV] = { 0 };
	float hessian[NV * NV] = { 0 };
	float t1, t2;

	float Nmax;
	float xc, yc, zc, wc;
	float xs, ys, zs, ws;
	float temp;
	float delta_f[256] = { 0 }, delta_dxf[256] = { 0 }, delta_dyf[256] = { 0 }, delta_dzf[256] = { 0 }, delta_dwf[256] = { 0 };
	int info;
	float L[NV * NV] = { 0 }, U[NV * NV] = { 0 };
	const float* initP = initP4;


	//Prevent read/write past end of array
	if (subregion >= Nfits) return;

	//for (ii=0;ii<spline_xsize*spline_ysize*spline_zsize;ii++)
	//       s_coeff[ii]=d_coeff[ii];

	 //if (threadIdx.x < 256) {
  //      for(int ii = threadIdx.x; ii  <spline_xsize*spline_ysize*spline_zsize; i += 256) {
  //          s_coeff[ii]=d_coeff[ii];
  //      }
  //  }
	 //__syncthreads();
	//__syncthread();


	for (ii = 0; ii < NV * NV; ii++)M[ii] = 0;
	for (ii = 0; ii < NV * NV; ii++)Minv[ii] = 0;

	//copy in data
	const float* s_data = d_data + (sz * sz * subregion);

	//const float *s_varim = d_varim+(sz*sz*bx*BlockSize+sz*sz*tx);
  //const float *s_gainim = d_gainim+(sz*sz*bx*BlockSize+sz*sz*tx);

  //initial values
	kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
	kernel_GaussFMaxMin2D(sz, 1.5, s_data, &Nmax, &newTheta[3]);
	newTheta[2] = max(0.0, (Nmax - newTheta[3]) * 2 * PI * 1.5 * 1.5);
	newTheta[3] = max(newTheta[3], 0.01);
	if (*initP!= 0)
	{
		newTheta[4] = *initP;
		newTheta[5] = *(initP +1);
	}
	else
	{
		newTheta[4] = float(spline_zsize) / 2.0f;
		newTheta[5] = float((spline_wsize)) / 2.0f;
	}
	//newClamp[2] = max(newTheta[2], newClamp[2]);
	//oldClamp[2] = newClamp[2];

	//newClamp[3] = max(newTheta[3], newClamp[3]);
	//oldClamp[3] = newClamp[3];

	//newClamp[4] = max(spline_zsize / 3.0f, newClamp[4]);
	//oldClamp[4] = newClamp[4];

	for (ii = 0; ii < NV; ii++)oldTheta[ii] = newTheta[ii];

	//updateFitValues3D

	//xc = -2.0*((newTheta[0]-float(sz)/2)+0.5);
	//yc = -2.0*((newTheta[1]-float(sz)/2)+0.5);

	//off = (float(spline_xsize)+1.0-2*float(sz))/2;

	//xstart = floor(xc);
	//xc = xc-xstart;

	//ystart = floor(yc);
	//yc = yc-ystart;

	////zstart = floor(newTheta[4]);
	//zstart = floor(newTheta[4]);
	//zc = newTheta[4] -zstart;

	//flipped in the bSpline
	xc = (sz + 1) / 2 - newTheta[0];
	yc = (sz + 1) / 2 - newTheta[1];
	zc = newTheta[4];
	wc = newTheta[5];
	//off = (spline_xsize - 4 - sz) / 2.0f;
	off = (spline_xsize - sz) / 2.0f;//new
	xs = xc - floor(xc);
	ys = yc - floor(yc);
	zs = zc - floor(zc);
	ws = wc - floor(wc);

	newErr = 0.0f;
	memset(jacobian, 0, NV * sizeof(float));
	memset(hessian, 0, NV * NV * sizeof(float));
	kernel_computeDelta4D_bSpline(xs, ys, zs, ws, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf);

	for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
		//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		kernel_Derivative_bSpline14D(xc + jj + off, yc + ii + off, zc, wc, spline_xsize, spline_ysize, spline_zsize, spline_wsize, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, s_coeff, newTheta, newDudt, &model);

		data = s_data[sz * jj + ii];
		newDudt[5] = newDudt[4];
		if (data > 0)
			newErr = newErr + 2 * ((model - data) - data * log(model / data));
		else
		{
			newErr = newErr + 2 * model;
			data = 0;
		}

		t1 = 1 - data / model;
		for (l = 0; l < NV; l++) {
			jacobian[l] += t1 * newDudt[l];
		}

		t2 = data / pow(model, 2);
		for (l = 0; l < NV; l++) for (m = l; m < NV; m++) {
			hessian[l * NV + m] += t2 * newDudt[l] * newDudt[m];
			hessian[m * NV + l] = hessian[l * NV + m];
		}
	}
	//addPeak

	//copyFitData

	for (kk = 0; kk < iterations; kk++) {//main iterative loop
	/*	if (kk >= 10)
			printf("now %d,%f,%f\n",kk, newTheta[4],newTheta[5]);*/


		if (fabs((newErr - oldErr) / newErr) < TOLERANCE) {
			//newStatus = CONVERGED;
			/*deg[90]=1;*/
			break;
		}
		else {
			if (newErr > 1.5 * oldErr) {
				//copy Fitdata

				for (i = 0; i < NV; i++) {
					newSign[i] = oldSign[i];
					newClamp[i] = oldClamp[i];
					newTheta[i] = oldTheta[i];
				}
				newLambda = oldLambda;
				newErr = oldErr;

				newLambda = 10 * newLambda;
			}
			else if (newErr < oldErr) {

				if (newLambda > 1) {
					newLambda = newLambda * 0.8;
				}
				else if (newLambda < 1) {
					newLambda = 1;
				}
			}


			for (i = 0; i < NV; i++) {
				hessian[i * NV + i] = hessian[i * NV + i] * newLambda;
			}
			memset(L, 0, NV * sizeof(float));
			memset(U, 0, NV * sizeof(float));

			info = kernel_cholesky(hessian, NV, L, U);
		loop1:
			if (info == 0) {
				kernel_luEvaluate(L, U, jacobian, NV, newUpdate);
				for (ll = 0; ll < 6; ll++)
				{
					if (isnan(newUpdate[ll]))
					{
						info = 1;
						goto loop1;
					}
				}
				newUpdate[5] = newUpdate[5] * 10;
				/*if (isnan(newUpdate))
				{
					info = 1; goto loop1;
				}*/
				//copyFitData
				for (i = 0; i < NV; i++) {
					oldSign[i] = newSign[i];
					oldClamp[i] = newClamp[i];

					oldTheta[i] = newTheta[i];
				}
				oldLambda = newLambda;
				oldErr = newErr;

				//let it go
				if (((fabs(newUpdate[4]) < 0.005) && turn))
				{
					newClamp[5] = float(spline_wsize) / 10; oldClamp[5] = float(spline_wsize) / 10; turn = 0;
					newClamp[4] = 2; oldClamp[4] = 2;
				}


				//updatePeakParameters
				for (ll = 0; ll < NV; ll++) {
					if (newSign[ll] != 0) {
						if (newSign[ll] == 1 && newUpdate[ll] < 0) {
							newClamp[ll] = newClamp[ll] * fade[ll];
						}
						else if (newSign[ll] == -1 && newUpdate[ll] > 0) {
							newClamp[ll] = newClamp[ll] * fade[ll];
						}
					}

					if (newUpdate[ll] > 0) {
						newSign[ll] = 1;
					}
					else {
						newSign[ll] = -1;
					}

					newTheta[ll] = newTheta[ll] - newUpdate[ll] / (1 + fabs(newUpdate[ll] / newClamp[ll]));
				}

				//newTheta[0] = max(newTheta[0], (float(sz) - 1) / 2 - sz / 4.0);
				//newTheta[0] = min(newTheta[0], (float(sz) - 1) / 2 + sz / 4.0);
				//newTheta[1] = max(newTheta[1], (float(sz) - 1) / 2 - sz / 4.0);
				//newTheta[1] = min(newTheta[1], (float(sz) - 1) / 2 + sz / 4.0);
				newTheta[2] = max(newTheta[2], 1.0);
				newTheta[3] = max(newTheta[3], 0.01);
				newTheta[4] = max(newTheta[4], 0.0);
				newTheta[4] = min(newTheta[4], float(spline_zsize) - 0.5);
				newTheta[5] = max(newTheta[5], 0.0);
				newTheta[5] = min(newTheta[5], float(spline_wsize) - 0.5);



				//updateFitValues3D
				//xc = -2.0*((newTheta[0]-float(sz)/2)+0.5);
				//yc = -2.0*((newTheta[1]-float(sz)/2)+0.5);

				//off = (float(spline_xsize)+1.0-2*float(sz))/2;

				//xstart = floor(xc);
				//xc = xc-xstart;

				//ystart = floor(yc);
				//yc = yc-ystart;

				////zstart = floor(newTheta[4]);
				//zstart = floor(newTheta[4]);
				//zc = newTheta[4] -zstart;
				//flipped in the bSpline
				xc = (sz + 1) / 2 - newTheta[0];
				yc = (sz + 1) / 2 - newTheta[1];
				zc = newTheta[4];
				wc = newTheta[5];
				//off = float(spline_xsize-4-sz)/2.0f;
				xs = xc - floor(xc);
				ys = yc - floor(yc);
				zs = zc - floor(zc);
				ws = wc - floor(wc);


				newErr = 0;
				memset(jacobian, 0, NV * sizeof(float));
				memset(hessian, 0, NV * NV * sizeof(float));
				kernel_computeDelta4D_bSpline(xs, ys, zs, ws, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf);

				for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
					//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
					kernel_Derivative_bSpline14D(xc + jj + off, yc + ii + off, zc, wc, spline_xsize, spline_ysize, spline_zsize, spline_wsize, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, s_coeff, newTheta, newDudt, &model);

					//temp = kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,s_coeff);
					//model = newTheta[3]+newTheta[2]*temp;
					data = s_data[sz * jj + ii];
					//calculating derivatives

					//newDudt[0] = -1*newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dxf,s_coeff);
					//newDudt[1] = -1*newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dyf,s_coeff);
					//newDudt[4] = newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dzf,s_coeff);
					//newDudt[2] = temp;
					//newDudt[3] = 1;

					if (data > 0)
						newErr = newErr + 2 * ((model - data) - data * log(model / data));
					else
					{
						newErr = newErr + 2 * model;
						data = 0;
					}

					t1 = 1 - data / model;
					for (l = 0; l < NV; l++) {
						jacobian[l] += t1 * newDudt[l];
					}

					t2 = data / pow(model, 2);
					for (l = 0; l < NV; l++) for (m = l; m < NV; m++) {
						hessian[l * NV + m] += t2 * newDudt[l] * newDudt[m];
						hessian[m * NV + l] = hessian[l * NV + m];
					}
				}
			}
			else
			{
				newLambda = 10 * newLambda;
			}

			//copyFitdata	

		}



	}
	d_Parameters[Nfits * NV + subregion] = kk;
	// Calculating the CRLB and LogLikelihood
	Div = 0.0;

	xc = (sz + 1) / 2 - newTheta[0];
	yc = (sz + 1) / 2 - newTheta[1];
	zc = newTheta[4];
	wc = newTheta[5];
	//off = float(spline_xsize-4-sz)/2.0f;
	xs = xc - floor(xc);
	ys = yc - floor(yc);
	zs = zc - floor(zc);
	ws = wc - floor(wc);

	kernel_computeDelta4D_bSpline(xs, ys, zs, ws, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf);




	for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
		kernel_Derivative_bSpline14D(xc + jj + off, yc + ii + off, zc, wc, spline_xsize, spline_ysize, spline_zsize, spline_wsize, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, s_coeff, newTheta, newDudt, &model);
		//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		data = s_data[sz * jj + ii];

		//Building the Fisher Information Matrix
		for (kk = 0; kk < NV; kk++)for (ll = kk; ll < NV; ll++) {
			M[kk * NV + ll] += newDudt[ll] * newDudt[kk] / model;
			M[ll * NV + kk] = M[kk * NV + ll];
		}

		//LogLikelyhood
		if (model > 0)
			if (data > 0)Div += data * log(model) - model - data * log(data) + data;
			else
				Div += -model;
	}

	// Matrix inverse (CRLB=F^-1) and output assigments
	kernel_MatInvN(M, Minv, Diag, NV);


	//write to global arrays
	for (kk = 0; kk < NV; kk++) d_Parameters[Nfits * kk + subregion] = newTheta[kk];
	for (kk = 0; kk < NV; kk++) d_CRLBs[Nfits * kk + subregion] = Diag[kk];
	d_LogLikelihood[subregion] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;


	return;
}



//****************/5D
void kernel_bsplineMLEFit_z_EMCCD_53D(const int subregion, const float* d_data, const float* d_coeff5,const float* d_coeff3, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int spline_wsize, const int spline_usize, const int sz, const int iterations, const float* initP5,
	float* d_Parameters, float* d_CRLBs, float* d_LogLikelihood, const int Nfits, float* deg) {

	//__shared__ float s_coeff[17*17*25];

	const int NV = NV_P53;
	const int NV3 = NV_PZ;

	float M[NV * NV], Diag[NV], Minv[NV * NV];
	//int tx = threadIdx.x;
	//int bx = blockIdx.x;
	//int BlockSize = blockDim.x;
	int ii, jj, kk, ll, l, m, i;
	int xstart, ystart, zstart, wstart, ustart, xi, yi, turn = 1;

	const float* s_coeff5;
	const float* s_coeff3;
	s_coeff5 = d_coeff5;
	s_coeff3 = d_coeff3;


	float model3, model5, modelAll, data;
	float Div;
	//float dudt[NV_PS];
	float newTheta[NV + 1], oldTheta[NV+1];
	float newLambda = 1.0, oldLambda = 1.0;
	float newSign[NV] = { 0 }, oldSign[NV] = { 0 };
	float newUpdate[NV] = { 0 }, oldUpdate[NV] = { 0 }, sumUpdate = 1e10;
	float newClamp[NV] = { 1.0,1.0,10000,20,2,2,2,0.1 }, oldClamp[NV];
	float newDudt[NV] = { 0 }, newDudt3[NV3] = { 0 };
	float  fade[NV] = { 0.5,0.5,0.5,0.5,0.5,0.5 ,0.5,0.5 };

	float newErr = 1e12, oldErr = 1e13;

	float off;
	float jacobian[NV] = { 0 };
	float hessian[NV * NV] = { 0 };
	float t1, t2;

	float Nmax;
	float xc, yc, zc, wc, uc;
	float xs, ys, zs, ws, us;
	float temp;
	float delta_f3[64] = { 0 }, delta_dxf3[64] = { 0 }, delta_dyf3[64] = { 0 }, delta_dzf3[64] = { 0 };
	float delta_f[1024] = { 0 }, delta_dxf[1024] = { 0 }, delta_dyf[1024] = { 0 }, delta_dzf[1024] = { 0 }, delta_dwf[1024] = { 0 }, delta_duf[1024] = { 0 };
	int info;
	float L[NV * NV] = { 0 }, U[NV * NV] = { 0 };
	float g2 = 0.9;
	const float* initP = initP5;

	//Prevent read/write past end of array
	if (subregion >= Nfits) return;

	//for (ii=0;ii<spline_xsize*spline_ysize*spline_zsize;ii++)
	//       s_coeff[ii]=d_coeff[ii];

	 //if (threadIdx.x < 256) {
  //      for(int ii = threadIdx.x; ii  <spline_xsize*spline_ysize*spline_zsize; i += 256) {
  //          s_coeff[ii]=d_coeff[ii];
  //      }
  //  }
	 //__syncthreads();
	//__syncthread();


	for (ii = 0; ii < NV * NV; ii++)M[ii] = 0;
	for (ii = 0; ii < NV * NV; ii++)Minv[ii] = 0;

	//copy in data
	const float* s_data = d_data + (sz * sz * subregion);

	//const float *s_varim = d_varim+(sz*sz*bx*BlockSize+sz*sz*tx);
  //const float *s_gainim = d_gainim+(sz*sz*bx*BlockSize+sz*sz*tx);

  //initial values
	kernel_CenterofMass2D(sz, s_data, &newTheta[0], &newTheta[1]);
	kernel_GaussFMaxMin2D(sz, 1.5, s_data, &Nmax, &newTheta[3]);
	newTheta[2] = max(0.0, (Nmax - newTheta[3]) * 2 * PI * 1.5 * 1.5);
	newTheta[3] = max(newTheta[3], 0.01);
	/*newTheta[4] = float(spline_zsize) / 2.0f;
	newTheta[5] = float((spline_wsize)) / 2.0f;
	newTheta[6] = float((spline_usize)) / 2.0f;*/

	if (initP5)
	{
		for (i = 4; i < NV; i++)
		{
			newTheta[i] = *(initP + i - 4);
		}
	}
	else
	{
		newTheta[4] = float(spline_zsize) / 2.0f;
		newTheta[5] = float((spline_wsize)) / 2.0f;
		newTheta[6] = float((spline_usize)) / 2.0f;
	}
	//newClamp[2] = max(newTheta[2], newClamp[2]);
	//oldClamp[2] = newClamp[2];

	//newClamp[3] = max(newTheta[3], newClamp[3]);
	//oldClamp[3] = newClamp[3];

	//newClamp[4] = max(spline_zsize / 3.0f, newClamp[4]);
	//oldClamp[4] = newClamp[4];

	for (ii = 0; ii < NV; ii++) {
		oldTheta[ii] = newTheta[ii];
		oldClamp[ii] = newClamp[ii];
	}
	//updateFitValues3D

	//xc = -2.0*((newTheta[0]-float(sz)/2)+0.5);
	//yc = -2.0*((newTheta[1]-float(sz)/2)+0.5);

	//off = (float(spline_xsize)+1.0-2*float(sz))/2;

	//xstart = floor(xc);
	//xc = xc-xstart;

	//ystart = floor(yc);
	//yc = yc-ystart;

	////zstart = floor(newTheta[4]);
	//zstart = floor(newTheta[4]);
	//zc = newTheta[4] -zstart;

	//flipped in the bSpline
	xc = (sz + 1) / 2 - newTheta[0];
	yc = (sz + 1) / 2 - newTheta[1];
	zc = newTheta[4];
	wc = newTheta[5];
	uc = newTheta[6];
	//off = (spline_xsize - 4 - sz) / 2.0f;
	off = (spline_xsize - sz) / 2.0f;//new
	xs = xc - floor(xc);
	ys = yc - floor(yc);
	zs = zc - floor(zc);
	ws = wc - floor(wc);
	us = uc - floor(uc);

	g2 = newTheta[7];

	newErr = 0.0f;
	memset(jacobian, 0, NV * sizeof(float));
	memset(hessian, 0, NV * NV * sizeof(float));
	kernel_computeDelta3D_bSpline(xs, ys, zs, delta_f3, delta_dxf3, delta_dyf3, delta_dzf3);
	kernel_computeDelta5D_bSpline(xs, ys, zs, ws, us, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, delta_duf);

	for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
		//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		kernel_Derivative_bSpline15D(xc + jj + off, yc + ii + off, zc, wc, uc, spline_xsize, spline_ysize, spline_zsize, spline_wsize, spline_usize, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, delta_duf, s_coeff5, newTheta, newDudt, &model5);
		kernel_Derivative_bSpline1(xc + jj + off, yc + ii + off, zc, spline_xsize, spline_ysize, spline_zsize, delta_f3, delta_dxf3, delta_dyf3, delta_dzf3, s_coeff3, newTheta, newDudt3, &model3);

		data = s_data[sz * jj + ii];

		for (l = 0; l < NV3; l++)newDudt[l] = newDudt[l] * g2 + (1 - g2) * newDudt3[l];

		modelAll = model3 * (1 - g2) + model5 * g2;

		newDudt[NV - 1] = model5 - model3;


		if (data > 0)
			newErr = newErr + 2 * ((modelAll - data) - data * log(modelAll / data));
		else
		{
			newErr = newErr + 2 * modelAll;
			data = 0;
		}

		t1 = 1 - data / modelAll;
		for (l = 0; l < NV; l++) {
			jacobian[l] += t1 * newDudt[l];
		}

		t2 = data / pow(modelAll, 2.0f);
		for (l = 0; l < NV; l++) for (m = l; m < NV; m++) {
			hessian[l * NV + m] += t2 * newDudt[l] * newDudt[m];
			hessian[m * NV + l] = hessian[l * NV + m];
		}

		
	}
	//addPeak

	//copyFitData

	for (kk = 0; kk < iterations; kk++) {//main iterative loop
	/*	if (kk >= 10)
			printf("now %d,%f,%f\n",kk, newTheta[4],newTheta[5]);*/


		if ((fabs((newErr - oldErr) / newErr) < TOLERANCE) || (sumUpdate < TOLERANCE1)) {
			//if ((fabs((newErr - oldErr) / newErr) < TOLERANCE) ) {
				//newStatus = CONVERGED;
				/*deg[90]=1;*/
			break;
		}
		else {
			if (newErr > 1.5 * oldErr) {
				//copy Fitdata

				for (i = 0; i < NV; i++) {
					newSign[i] = oldSign[i];
					newClamp[i] = oldClamp[i];
					newTheta[i] = oldTheta[i];
				}
				newLambda = oldLambda;
				newErr = oldErr;

				newLambda = 10 * newLambda;
			}
			else if (newErr < oldErr) {

				if (newLambda > 1) {
					newLambda = newLambda * 0.8;
				}
				else if (newLambda < 1) {
					newLambda = 1;
				}
			}


			for (i = 0; i < NV; i++) {
				hessian[i * NV + i] = hessian[i * NV + i] * newLambda;
			}
			memset(L, 0, NV * sizeof(float));
			memset(U, 0, NV * sizeof(float));

			info = kernel_cholesky(hessian, NV, L, U);
		loop1:
			if (info == 0) {
				kernel_luEvaluate5D(L, U, jacobian, NV, newUpdate);
				for (ll = 0; ll < NV; ll++)
				{
					if (isnan(newUpdate[ll]))
					{
						info = 1;
						goto loop1;
					}
				}

				/*if (isnan(newUpdate))
				{
					info = 1; goto loop1;
				}*/
				//copyFitData
				for (i = 0; i < NV; i++) {
					oldSign[i] = newSign[i];
					oldClamp[i] = newClamp[i];

					oldTheta[i] = newTheta[i];
				}
				oldLambda = newLambda;
				oldErr = newErr;

				//let it go
				/*if (((fabs(newUpdate[4]) < 0.005) && turn))
				{
					newClamp[5] = float(spline_wsize) / 10; oldClamp[5] = float(spline_wsize) / 10; turn = 0;
					newClamp[4] = 2; oldClamp[4] = 2;
				}*/

				sumUpdate = 0;
				//updatePeakParameters
				for (ll = 0; ll < NV; ll++) {
					if (newSign[ll] != 0) {
						if (newSign[ll] == 1 && newUpdate[ll] < 0) {
							newClamp[ll] = newClamp[ll] * fade[ll];
						}
						else if (newSign[ll] == -1 && newUpdate[ll] > 0) {
							newClamp[ll] = newClamp[ll] * fade[ll];
						}
					}

					if (newUpdate[ll] > 0) {
						newSign[ll] = 1;
					}
					else {
						newSign[ll] = -1;
					}
					newUpdate[ll] = newUpdate[ll] / (1 + fabs(newUpdate[ll] / newClamp[ll]));
					newTheta[ll] = newTheta[ll] - newUpdate[ll];
					sumUpdate += fabs(newUpdate[ll]);
				}

				newTheta[0] = max(newTheta[0], (float(sz) - 1) / 2 - sz / 4.0);
				newTheta[0] = min(newTheta[0], (float(sz) - 1) / 2 + sz / 4.0);
				newTheta[1] = max(newTheta[1], (float(sz) - 1) / 2 - sz / 4.0);
				newTheta[1] = min(newTheta[1], (float(sz) - 1) / 2 + sz / 4.0);

				newTheta[2] = max(newTheta[2], 1.0);
				newTheta[3] = max(newTheta[3], 0.01);
				newTheta[4] = max(newTheta[4], 0.0);
				newTheta[4] = min(newTheta[4], float(spline_zsize) - 0.5);
				/*newTheta[5] = max(newTheta[5], 0.0);
				newTheta[5] = min(newTheta[5], float(spline_wsize) - 0.5);*/
				if (newTheta[5] <= 1)
				{
					newTheta[5] = spline_wsize - 2 + newTheta[5];
				}
				else if (newTheta[5] > (spline_wsize - 1))
				{
					newTheta[5] = 2 + newTheta[5] - spline_wsize;
				}
				newTheta[6] = max(newTheta[6], 1);
				newTheta[6] = min(newTheta[6], float(spline_usize) - 0.5);

				newTheta[7] = min(newTheta[7], 1);// ratio
				newTheta[7] = max(newTheta[7], 0);

				//updateFitValues3D
				//xc = -2.0*((newTheta[0]-float(sz)/2)+0.5);
				//yc = -2.0*((newTheta[1]-float(sz)/2)+0.5);

				//off = (float(spline_xsize)+1.0-2*float(sz))/2;

				//xstart = floor(xc);
				//xc = xc-xstart;

				//ystart = floor(yc);
				//yc = yc-ystart;

				////zstart = floor(newTheta[4]);
				//zstart = floor(newTheta[4]);
				//zc = newTheta[4] -zstart;
				//flipped in the bSpline
				xc = (sz + 1) / 2 - newTheta[0];
				yc = (sz + 1) / 2 - newTheta[1];
				zc = newTheta[4];
				wc = newTheta[5];
				uc = newTheta[6];
				//off = float(spline_xsize-4-sz)/2.0f;
				xs = xc - floor(xc);
				ys = yc - floor(yc);
				zs = zc - floor(zc);
				ws = wc - floor(wc);
				us = uc - floor(uc);

				g2 = newTheta[7];

				newErr = 0;
				memset(jacobian, 0, NV * sizeof(float));
				memset(hessian, 0, NV * NV * sizeof(float));
				kernel_computeDelta3D_bSpline(xs, ys, zs, delta_f3, delta_dxf3, delta_dyf3, delta_dzf3);
				kernel_computeDelta5D_bSpline(xs, ys, zs, ws, us, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, delta_duf);

				for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
					//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
					kernel_Derivative_bSpline15D(xc + jj + off, yc + ii + off, zc, wc, uc, spline_xsize, spline_ysize, spline_zsize, spline_wsize, spline_usize, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, delta_duf, s_coeff5, newTheta, newDudt, &model5);
					kernel_Derivative_bSpline1(xc + jj + off, yc + ii + off, zc, spline_xsize, spline_ysize, spline_zsize, delta_f3, delta_dxf3, delta_dyf3, delta_dzf3, s_coeff3, newTheta, newDudt3, &model3);

					//temp = kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,s_coeff);
					//model = newTheta[3]+newTheta[2]*temp;
					data = s_data[sz * jj + ii];
					//calculating derivatives

					//newDudt[0] = -1*newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dxf,s_coeff);
					//newDudt[1] = -1*newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dyf,s_coeff);
					//newDudt[4] = newTheta[2]*kernal_fAt3D(2*ii+xstart+off,2*jj+ystart+off,zstart, spline_xsize,spline_ysize,spline_zsize,delta_dzf,s_coeff);
					//newDudt[2] = temp;
					//newDudt[3] = 1;

					modelAll = model3 * (1 - g2) + model5 * g2;

					for (l = 0; l < NV3; l++)newDudt[l] = newDudt[l] * g2 + (1 - g2) * newDudt3[l];

					newDudt[NV - 1] = model5 - model3;

					if (data > 0)
						newErr = newErr + 2 * ((modelAll - data) - data * log(modelAll / data));
					else
					{
						newErr = newErr + 2 * modelAll;
						data = 0;
					}

					t1 = 1 - data / modelAll;
					for (l = 0; l < NV; l++) {
						jacobian[l] += t1 * newDudt[l];
					}

					t2 = data / pow(modelAll, 2);
					for (l = 0; l < NV; l++) for (m = l; m < NV; m++) {
						hessian[l * NV + m] += t2 * newDudt[l] * newDudt[m];
						hessian[m * NV + l] = hessian[l * NV + m];
					}
				}
			}
			else
			{
				newLambda = 10 * newLambda;
			}

			//copyFitdata	

		}



	}
	newTheta[NV] = kk;
	// Calculating the CRLB and LogLikelihood
	Div = 0.0;

	//xc = (sz + 1) / 2 - newTheta[0];
	//yc = (sz + 1) / 2 - newTheta[1];
	//zc = newTheta[4];
	//wc = newTheta[5];
	//uc = newTheta[6];
	////off = float(spline_xsize-4-sz)/2.0f;
	//xs = xc - floor(xc);
	//ys = yc - floor(yc);
	//zs = zc - floor(zc);
	//ws = wc - floor(wc);
	//us = uc - floor(uc);

	//kernel_computeDelta5D_bSpline(xs, ys, zs, ws, us, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, delta_duf);


	for (ii = 0; ii < sz; ii++) for (jj = 0; jj < sz; jj++) {
		kernel_Derivative_bSpline15D(xc + jj + off, yc + ii + off, zc, wc, uc, spline_xsize, spline_ysize, spline_zsize, spline_wsize, spline_usize, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_dwf, delta_duf, s_coeff5, newTheta, newDudt, &model5);
		//kernel_DerivativeSpline(2*ii+xstart+off,2*jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,s_coeff,newTheta,newDudt,&model);
		kernel_Derivative_bSpline1(xc + jj + off, yc + ii + off, zc, spline_xsize, spline_ysize, spline_zsize, delta_f3, delta_dxf3, delta_dyf3, delta_dzf3, s_coeff3, newTheta, newDudt3, &model3);

		data = s_data[sz * jj + ii];

		for (l = 0; l < NV3; l++) newDudt[l] = newDudt[l] * g2 + (1 - g2) * newDudt3[l];

		modelAll = model3 * (1 - g2) + model5 * g2;
		newDudt[NV - 1] = model5 - model3;

		//Building the Fisher Information Matrix
		for (kk = 0; kk < NV; kk++)for (ll = kk; ll < NV; ll++) {
			M[kk * NV + ll] += newDudt[ll] * newDudt[kk] / modelAll;
			M[ll * NV + kk] = M[kk * NV + ll];
		}

		//LogLikelyhood
		if (modelAll > 0)
			if (data > 0)Div += data * log(modelAll) - modelAll - data * log(data) + data;
			else
				Div += -modelAll;
	}

	// Matrix inverse (CRLB=F^-1) and output assigments
	kernel_MatInvN(M, Minv, Diag, NV);


	//write to global arrays
	for (kk = 0; kk < NV + 1; kk++) d_Parameters[Nfits * kk + subregion] = newTheta[kk];
	for (kk = 0; kk < NV; kk++) d_CRLBs[Nfits * kk + subregion] = Diag[kk];
	d_LogLikelihood[subregion] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;


	return;
}

