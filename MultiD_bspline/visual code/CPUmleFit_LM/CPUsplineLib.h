#ifndef CPUSPLINELIB_H
#define CPUSPLINELIB_H


//5D

void kernel_luEvaluate5D(float* L, float* U, float* b, const int n, float* x);

void kernel_computeDelta5D_bSpline(float x_delta, float y_delta, float z_delta, float w_delta, float u_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, float* delta_dwf, float* delta_duf);

void kernel_Derivative_bSpline15D(float xi, float yi, float zi, float wi, float ui, int nDatax, int nDatay, int nDataz, int nDataw, int nDatau, float* delta_f, float* delta_dfx, float* delta_dfy, float* delta_dfz, float* delta_dfw, float* delta_dfu, const float* cMat, float* theta, float* dudt, float* model);

//4D
void kernel_computeDelta4D_bSpline(float x_delta, float y_delta, float z_delta, float w_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, float* delta_dwf);

void kernel_Derivative_bSpline14D(float xi, float yi, float zi, float wi, int nDatax, int nDatay, int nDataz, int nDataw, float* delta_f, float* delta_dfx, float* delta_dfy, float* delta_dfz, float* delta_dfw, const float* cMat, float* theta, float* dudt, float* model);

//3D
void kernel_computeDelta3D_bSpline(float x_delta, float y_delta, float z_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf);
void kernel_Derivative_bSpline1(float xi, float yi, float zi, int nDatax, int nDatay, int nDataz, float* delta_f, float* delta_dfx, float* delta_dfy, float* delta_dfz, const float* cMat, float* theta, float* dudt, float* model);


//**************************************************************************************************************************************************



 void kernel_computeDelta3D(float x_delta, float y_delta, float z_delta, float *delta_f, float *delta_dxf, float *delta_dyf, float *delta_dzf);

 int kernel_cholesky(float *A,int n, float *L, float*U);

 void kernel_luEvaluate(float *L,float *U, float *b, int n, float *x);


 void kernel_DerivativeSpline(int xc, int yc, int zc, int xsize, int ysize, int zsize, float *delta_f, float *delta_dxf, float *delta_dyf, float *delta_dzf,const float *coeff,float *theta, float*dudt,float*model);

 void kernel_evalBSpline(float xi, int deg, float* bi);


#endif