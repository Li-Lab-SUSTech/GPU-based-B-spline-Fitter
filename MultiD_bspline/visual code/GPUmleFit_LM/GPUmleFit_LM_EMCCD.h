/*!
 * \file GPUmleFit_LM_EMCCD.h
 //author Yiming Li
//date 2023.09.22
 * \brief Prototypes for the actual Cuda kernels.  
 */
#include "definitions.h"
#ifndef GPUMLEFIT_LM_EMCCD_H
#define GPUMLEFIT_LM_EMCCD_H
__global__ void kernel_MLEFit_LM_EMCCD(const float *d_data,const float PSFSigma, const int sz, const int iterations, 
	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits);

__global__ void kernel_MLEFit_LM_Sigma_EMCCD(const float *d_data,const float PSFSigma, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits);

__global__ void kernel_MLEFit_LM_z_EMCCD(const float *d_data, const float PSFSigma_x, const float Ax, const float Ay, const float Bx, 
	const float By, const float gamma, const float d, const float PSFSigma_y, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits);

__global__ void kernel_MLEFit_LM_sigmaxy_EMCCD(const float *d_data, const float PSFSigma, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,const int Nfits);

__global__ void kernel_splineMLEFit_z_EMCCD(const float *d_data,const float *d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int sz, const int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,float initZ, const int Nfits);

__global__ void kernel_bsplineMLEFit_z_EMCCD(const float* d_data, const float* d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int sz, const int iterations,
	float* d_Parameters, float* d_CRLBs, float* d_LogLikelihood, const int Nfits, float* deg, float* tmp);


__global__ void kernel_bsplineMLEFit_4D_EMCCD(const float* d_data, const float* d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int spline_wsize, const int sz, const int iterations, const float* initP4,
	float* d_Parameters, float* d_CRLBs, float* d_LogLikelihood, const int Nfits, float* deg);

__global__ void kernel_bsplineMLEFit_5n3D_EMCCD(const float* d_data, const float* d_coeff5, const float* d_coeff3, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int spline_wsize, const int spline_usize, const int sz, const int iterations, const float* initP5,
	float* d_Parameters, float* d_CRLBs, float* d_LogLikelihood, const int Nfits, float* deg);



#endif