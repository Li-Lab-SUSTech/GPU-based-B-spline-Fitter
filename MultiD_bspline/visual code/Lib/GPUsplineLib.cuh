/*!
 * \file GPUmleFit_LM_sCMOS.h
 //author Yiming Li
//date 20170301
*/
#include "GPUsplineLib.h"
#include "definitions.h"
//#define pi 3.141592f



//**************************************************************************************************************************************************
// This function for calculation of the common term for Cspline is adpopted from
//"Analyzing Single Molecule Localization Microscopy Data Using Cubic Splines", Hazen Babcok, Xiaowei Zhuang,Scientific Report, 1, 552 , 2017.

//5D
__device__ inline void kernel_computeDelta5D_bSpline(float x_delta, float y_delta, float z_delta, float w_delta, float u_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, float* delta_dwf, float* delta_duf) {
	float dx_delta, dy_delta, dz_delta, dw_delta, du_delta;
	int nIndex = 0;
	int q, j, i, r, s;
	float Bu1, Bu2, dBu1, dBu2, Bw1, Bw2, dBw1, dBw2, Bz1, Bz2, dBz1, dBz2, Bx1, Bx2, dBx1, dBx2, By1, By2, dBy1, dBy2;

	dx_delta = x_delta - 1.0f;
	dy_delta = y_delta - 1.0f;
	dz_delta = z_delta - 1.0f;
	dw_delta = w_delta - 1.0f;
	du_delta = u_delta - 1.0f;

	for (s = 1; s <= 2; s++) {
		kernel_evalBSpline(u_delta + s - 1.0f, 3, &Bu1);
		kernel_evalBSpline(u_delta - s, 3, &Bu2);
		kernel_evalBSpline(du_delta + s - 1.0f / 2.0f, 2, &dBu1);
		kernel_evalBSpline(du_delta - s + 1.0f / 2.0f, 2, &dBu2);

		for (r = 1; r <= 2; r++) {
			kernel_evalBSpline(w_delta + r - 1.0f, 3, &Bw1);
			kernel_evalBSpline(w_delta - r, 3, &Bw2);
			kernel_evalBSpline(dw_delta + r - 1.0f / 2.0f, 2, &dBw1);
			kernel_evalBSpline(dw_delta - r + 1.0f / 2.0f, 2, &dBw2);
			for (q = 1; q <= 2; q++) {
				kernel_evalBSpline(z_delta + q - 1.0f, 3, &Bz1);
				kernel_evalBSpline(z_delta - q, 3, &Bz2);
				kernel_evalBSpline(dz_delta + q - 1.0f / 2.0f, 2, &dBz1);
				kernel_evalBSpline(dz_delta - q + 1.0f / 2.0f, 2, &dBz2);

				for (j = 1; j <= 2; j++) {
					kernel_evalBSpline(x_delta + j - 1.0f, 3, &Bx1);
					kernel_evalBSpline(x_delta - j, 3, &Bx2);
					kernel_evalBSpline(dx_delta + j - 1.0f / 2.0f, 2, &dBx1);
					kernel_evalBSpline(dx_delta - j + 1.0f / 2.0f, 2, &dBx2);
					for (i = 1; i <= 2; i++) {
						kernel_evalBSpline(y_delta + i - 1.0f, 3, &By1);
						kernel_evalBSpline(y_delta - i, 3, &By2);
						kernel_evalBSpline(dy_delta + i - 1.0f / 2.0f, 2, &dBy1);
						kernel_evalBSpline(dy_delta - i + 1.0f / 2.0f, 2, &dBy2);

						delta_f[32 * nIndex] = By1 * Bx1 * Bz1 * Bw1 * Bu1;
						delta_f[32 * nIndex + 1] = By2 * Bx1 * Bz1 * Bw1 * Bu1;
						delta_f[32 * nIndex + 2] = By1 * Bx2 * Bz1 * Bw1 * Bu1;
						delta_f[32 * nIndex + 3] = By2 * Bx2 * Bz1 * Bw1 * Bu1;
						delta_f[32 * nIndex + 4] = By1 * Bx1 * Bz2 * Bw1 * Bu1;
						delta_f[32 * nIndex + 5] = By2 * Bx1 * Bz2 * Bw1 * Bu1;
						delta_f[32 * nIndex + 6] = By1 * Bx2 * Bz2 * Bw1 * Bu1;
						delta_f[32 * nIndex + 7] = By2 * Bx2 * Bz2 * Bw1 * Bu1;
						delta_f[32 * nIndex + 8] = By1 * Bx1 * Bz1 * Bw2 * Bu1;
						delta_f[32 * nIndex + 9] = By2 * Bx1 * Bz1 * Bw2 * Bu1;
						delta_f[32 * nIndex + 10] = By1 * Bx2 * Bz1 * Bw2 * Bu1;
						delta_f[32 * nIndex + 11] = By2 * Bx2 * Bz1 * Bw2 * Bu1;
						delta_f[32 * nIndex + 12] = By1 * Bx1 * Bz2 * Bw2 * Bu1;
						delta_f[32 * nIndex + 13] = By2 * Bx1 * Bz2 * Bw2 * Bu1;
						delta_f[32 * nIndex + 14] = By1 * Bx2 * Bz2 * Bw2 * Bu1;
						delta_f[32 * nIndex + 15] = By2 * Bx2 * Bz2 * Bw2 * Bu1;

						delta_f[32 * nIndex + 16] = By1 * Bx1 * Bz1 * Bw1 * Bu2;
						delta_f[32 * nIndex + 17] = By2 * Bx1 * Bz1 * Bw1 * Bu2;
						delta_f[32 * nIndex + 18] = By1 * Bx2 * Bz1 * Bw1 * Bu2;
						delta_f[32 * nIndex + 19] = By2 * Bx2 * Bz1 * Bw1 * Bu2;
						delta_f[32 * nIndex + 20] = By1 * Bx1 * Bz2 * Bw1 * Bu2;
						delta_f[32 * nIndex + 21] = By2 * Bx1 * Bz2 * Bw1 * Bu2;
						delta_f[32 * nIndex + 22] = By1 * Bx2 * Bz2 * Bw1 * Bu2;
						delta_f[32 * nIndex + 23] = By2 * Bx2 * Bz2 * Bw1 * Bu2;
						delta_f[32 * nIndex + 24] = By1 * Bx1 * Bz1 * Bw2 * Bu2;
						delta_f[32 * nIndex + 25] = By2 * Bx1 * Bz1 * Bw2 * Bu2;
						delta_f[32 * nIndex + 26] = By1 * Bx2 * Bz1 * Bw2 * Bu2;
						delta_f[32 * nIndex + 27] = By2 * Bx2 * Bz1 * Bw2 * Bu2;
						delta_f[32 * nIndex + 28] = By1 * Bx1 * Bz2 * Bw2 * Bu2;
						delta_f[32 * nIndex + 29] = By2 * Bx1 * Bz2 * Bw2 * Bu2;
						delta_f[32 * nIndex + 30] = By1 * Bx2 * Bz2 * Bw2 * Bu2;
						delta_f[32 * nIndex + 31] = By2 * Bx2 * Bz2 * Bw2 * Bu2;


						//dx
						delta_dxf[32 * nIndex] = dBy1 * Bx1 * Bz1 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 1] = dBy2 * Bx1 * Bz1 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 2] = dBy1 * Bx2 * Bz1 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 3] = dBy2 * Bx2 * Bz1 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 4] = dBy1 * Bx1 * Bz2 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 5] = dBy2 * Bx1 * Bz2 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 6] = dBy1 * Bx2 * Bz2 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 7] = dBy2 * Bx2 * Bz2 * Bw1 * Bu1;
						delta_dxf[32 * nIndex + 8] = dBy1 * Bx1 * Bz1 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 9] = dBy2 * Bx1 * Bz1 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 10] = dBy1 * Bx2 * Bz1 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 11] = dBy2 * Bx2 * Bz1 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 12] = dBy1 * Bx1 * Bz2 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 13] = dBy2 * Bx1 * Bz2 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 14] = dBy1 * Bx2 * Bz2 * Bw2 * Bu1;
						delta_dxf[32 * nIndex + 15] = dBy2 * Bx2 * Bz2 * Bw2 * Bu1;

						delta_dxf[32 * nIndex + 16] = dBy1 * Bx1 * Bz1 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 17] = dBy2 * Bx1 * Bz1 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 18] = dBy1 * Bx2 * Bz1 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 19] = dBy2 * Bx2 * Bz1 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 20] = dBy1 * Bx1 * Bz2 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 21] = dBy2 * Bx1 * Bz2 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 22] = dBy1 * Bx2 * Bz2 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 23] = dBy2 * Bx2 * Bz2 * Bw1 * Bu2;
						delta_dxf[32 * nIndex + 24] = dBy1 * Bx1 * Bz1 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 25] = dBy2 * Bx1 * Bz1 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 26] = dBy1 * Bx2 * Bz1 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 27] = dBy2 * Bx2 * Bz1 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 28] = dBy1 * Bx1 * Bz2 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 29] = dBy2 * Bx1 * Bz2 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 30] = dBy1 * Bx2 * Bz2 * Bw2 * Bu2;
						delta_dxf[32 * nIndex + 31] = dBy2 * Bx2 * Bz2 * Bw2 * Bu2;
						//dy
						delta_dyf[32 * nIndex] = By1 * dBx1 * Bz1 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 1] = By2 * dBx1 * Bz1 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 2] = By1 * dBx2 * Bz1 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 3] = By2 * dBx2 * Bz1 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 4] = By1 * dBx1 * Bz2 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 5] = By2 * dBx1 * Bz2 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 6] = By1 * dBx2 * Bz2 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 7] = By2 * dBx2 * Bz2 * Bw1 * Bu1;
						delta_dyf[32 * nIndex + 8] = By1 * dBx1 * Bz1 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 9] = By2 * dBx1 * Bz1 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 10] = By1 * dBx2 * Bz1 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 11] = By2 * dBx2 * Bz1 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 12] = By1 * dBx1 * Bz2 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 13] = By2 * dBx1 * Bz2 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 14] = By1 * dBx2 * Bz2 * Bw2 * Bu1;
						delta_dyf[32 * nIndex + 15] = By2 * dBx2 * Bz2 * Bw2 * Bu1;

						delta_dyf[32 * nIndex + 16] = By1 * dBx1 * Bz1 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 17] = By2 * dBx1 * Bz1 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 18] = By1 * dBx2 * Bz1 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 19] = By2 * dBx2 * Bz1 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 20] = By1 * dBx1 * Bz2 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 21] = By2 * dBx1 * Bz2 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 22] = By1 * dBx2 * Bz2 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 23] = By2 * dBx2 * Bz2 * Bw1 * Bu2;
						delta_dyf[32 * nIndex + 24] = By1 * dBx1 * Bz1 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 25] = By2 * dBx1 * Bz1 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 26] = By1 * dBx2 * Bz1 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 27] = By2 * dBx2 * Bz1 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 28] = By1 * dBx1 * Bz2 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 29] = By2 * dBx1 * Bz2 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 30] = By1 * dBx2 * Bz2 * Bw2 * Bu2;
						delta_dyf[32 * nIndex + 31] = By2 * dBx2 * Bz2 * Bw2 * Bu2;

						//dz
						delta_dzf[32 * nIndex] = By1 * Bx1 * dBz1 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 1] = By2 * Bx1 * dBz1 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 2] = By1 * Bx2 * dBz1 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 3] = By2 * Bx2 * dBz1 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 4] = By1 * Bx1 * dBz2 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 5] = By2 * Bx1 * dBz2 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 6] = By1 * Bx2 * dBz2 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 7] = By2 * Bx2 * dBz2 * Bw1 * Bu1;
						delta_dzf[32 * nIndex + 8] = By1 * Bx1 * dBz1 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 9] = By2 * Bx1 * dBz1 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 10] = By1 * Bx2 * dBz1 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 11] = By2 * Bx2 * dBz1 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 12] = By1 * Bx1 * dBz2 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 13] = By2 * Bx1 * dBz2 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 14] = By1 * Bx2 * dBz2 * Bw2 * Bu1;
						delta_dzf[32 * nIndex + 15] = By2 * Bx2 * dBz2 * Bw2 * Bu1;

						delta_dzf[32 * nIndex + 16] = By1 * Bx1 * dBz1 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 17] = By2 * Bx1 * dBz1 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 18] = By1 * Bx2 * dBz1 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 19] = By2 * Bx2 * dBz1 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 20] = By1 * Bx1 * dBz2 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 21] = By2 * Bx1 * dBz2 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 22] = By1 * Bx2 * dBz2 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 23] = By2 * Bx2 * dBz2 * Bw1 * Bu2;
						delta_dzf[32 * nIndex + 24] = By1 * Bx1 * dBz1 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 25] = By2 * Bx1 * dBz1 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 26] = By1 * Bx2 * dBz1 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 27] = By2 * Bx2 * dBz1 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 28] = By1 * Bx1 * dBz2 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 29] = By2 * Bx1 * dBz2 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 30] = By1 * Bx2 * dBz2 * Bw2 * Bu2;
						delta_dzf[32 * nIndex + 31] = By2 * Bx2 * dBz2 * Bw2 * Bu2;
						//dw
						delta_dwf[32 * nIndex] = By1 * Bx1 * Bz1 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 1] = By2 * Bx1 * Bz1 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 2] = By1 * Bx2 * Bz1 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 3] = By2 * Bx2 * Bz1 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 4] = By1 * Bx1 * Bz2 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 5] = By2 * Bx1 * Bz2 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 6] = By1 * Bx2 * Bz2 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 7] = By2 * Bx2 * Bz2 * dBw1 * Bu1;
						delta_dwf[32 * nIndex + 8] = By1 * Bx1 * Bz1 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 9] = By2 * Bx1 * Bz1 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 10] = By1 * Bx2 * Bz1 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 11] = By2 * Bx2 * Bz1 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 12] = By1 * Bx1 * Bz2 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 13] = By2 * Bx1 * Bz2 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 14] = By1 * Bx2 * Bz2 * dBw2 * Bu1;
						delta_dwf[32 * nIndex + 15] = By2 * Bx2 * Bz2 * dBw2 * Bu1;

						delta_dwf[32 * nIndex + 16] = By1 * Bx1 * Bz1 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 17] = By2 * Bx1 * Bz1 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 18] = By1 * Bx2 * Bz1 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 19] = By2 * Bx2 * Bz1 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 20] = By1 * Bx1 * Bz2 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 21] = By2 * Bx1 * Bz2 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 22] = By1 * Bx2 * Bz2 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 23] = By2 * Bx2 * Bz2 * dBw1 * Bu2;
						delta_dwf[32 * nIndex + 24] = By1 * Bx1 * Bz1 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 25] = By2 * Bx1 * Bz1 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 26] = By1 * Bx2 * Bz1 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 27] = By2 * Bx2 * Bz1 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 28] = By1 * Bx1 * Bz2 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 29] = By2 * Bx1 * Bz2 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 30] = By1 * Bx2 * Bz2 * dBw2 * Bu2;
						delta_dwf[32 * nIndex + 31] = By2 * Bx2 * Bz2 * dBw2 * Bu2;

						//du
						delta_duf[32 * nIndex] = By1 * Bx1 * Bz1 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 1] = By2 * Bx1 * Bz1 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 2] = By1 * Bx2 * Bz1 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 3] = By2 * Bx2 * Bz1 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 4] = By1 * Bx1 * Bz2 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 5] = By2 * Bx1 * Bz2 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 6] = By1 * Bx2 * Bz2 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 7] = By2 * Bx2 * Bz2 * Bw1 * dBu1;
						delta_duf[32 * nIndex + 8] = By1 * Bx1 * Bz1 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 9] = By2 * Bx1 * Bz1 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 10] = By1 * Bx2 * Bz1 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 11] = By2 * Bx2 * Bz1 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 12] = By1 * Bx1 * Bz2 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 13] = By2 * Bx1 * Bz2 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 14] = By1 * Bx2 * Bz2 * Bw2 * dBu1;
						delta_duf[32 * nIndex + 15] = By2 * Bx2 * Bz2 * Bw2 * dBu1;

						delta_duf[32 * nIndex + 16] = By1 * Bx1 * Bz1 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 17] = By2 * Bx1 * Bz1 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 18] = By1 * Bx2 * Bz1 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 19] = By2 * Bx2 * Bz1 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 20] = By1 * Bx1 * Bz2 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 21] = By2 * Bx1 * Bz2 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 22] = By1 * Bx2 * Bz2 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 23] = By2 * Bx2 * Bz2 * Bw1 * dBu2;
						delta_duf[32 * nIndex + 24] = By1 * Bx1 * Bz1 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 25] = By2 * Bx1 * Bz1 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 26] = By1 * Bx2 * Bz1 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 27] = By2 * Bx2 * Bz1 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 28] = By1 * Bx1 * Bz2 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 29] = By2 * Bx1 * Bz2 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 30] = By1 * Bx2 * Bz2 * Bw2 * dBu2;
						delta_duf[32 * nIndex + 31] = By2 * Bx2 * Bz2 * Bw2 * dBu2;

						/* delta_dxf[8 * nIndex] = dBy1 * Bx1 * Bz1;
						 delta_dxf[8 * nIndex + 1] = dBy2 * Bx1 * Bz1;
						 delta_dxf[8 * nIndex + 2] = dBy1 * Bx2 * Bz1;
						 delta_dxf[8 * nIndex + 3] = dBy2 * Bx2 * Bz1;
						 delta_dxf[8 * nIndex + 4] = dBy1 * Bx1 * Bz2;
						 delta_dxf[8 * nIndex + 5] = dBy2 * Bx1 * Bz2;
						 delta_dxf[8 * nIndex + 6] = dBy1 * Bx2 * Bz2;
						 delta_dxf[8 * nIndex + 7] = dBy2 * Bx2 * Bz2;

						 delta_dyf[8 * nIndex] = By1 * dBx1 * Bz1;
						 delta_dyf[8 * nIndex + 1] = By2 * dBx1 * Bz1;
						 delta_dyf[8 * nIndex + 2] = By1 * dBx2 * Bz1;
						 delta_dyf[8 * nIndex + 3] = By2 * dBx2 * Bz1;
						 delta_dyf[8 * nIndex + 4] = By1 * dBx1 * Bz2;
						 delta_dyf[8 * nIndex + 5] = By2 * dBx1 * Bz2;
						 delta_dyf[8 * nIndex + 6] = By1 * dBx2 * Bz2;
						 delta_dyf[8 * nIndex + 7] = By2 * dBx2 * Bz2;

						 delta_dzf[8 * nIndex] = By1 * Bx1 * dBz1;
						 delta_dzf[8 * nIndex + 1] = By2 * Bx1 * dBz1;
						 delta_dzf[8 * nIndex + 2] = By1 * Bx2 * dBz1;
						 delta_dzf[8 * nIndex + 3] = By2 * Bx2 * dBz1;
						 delta_dzf[8 * nIndex + 4] = By1 * Bx1 * dBz2;
						 delta_dzf[8 * nIndex + 5] = By2 * Bx1 * dBz2;
						 delta_dzf[8 * nIndex + 6] = By1 * Bx2 * dBz2;
						 delta_dzf[8 * nIndex + 7] = By2 * Bx2 * dBz2;*/

						nIndex = nIndex + 1;
					}
				}
			}
		}
	}
}
//**************************************************************************************************************



__device__ inline void kernel_Derivative_bSpline15D(float xi, float yi, float zi, float wi, float ui, int nDatax, int nDatay, int nDataz, int nDataw, int nDatau, float* delta_f, float* delta_dfx, float* delta_dfy, float* delta_dfz, float* delta_dfw, float* delta_dfu, const float* cMat, float* theta, float* dudt, float* model) {
	int xc, yc, zc, wc, uc, kx, ky, kz, kw, ku, q, j, i, r, s;
	int nIndex = 0.0f;
	float f = 0.0f, dfx = 0.0f, dfy = 0.0f, dfz = 0.0f, dfw = 0.0f, dfu = 0.0f;

	xc = floor(xi);
	yc = floor(yi);
	zc = floor(zi);
	wc = floor(wi);
	uc = floor(ui);

	xc = max(xc, 1);
	xc = min(xc, nDatax - 1);

	yc = max(yc, 1);
	yc = min(yc, nDatay - 1);

	zc = max(zc, 1);
	zc = min(zc, nDataz - 1);

	wc = max(wc, 1);
	wc = min(wc, nDataw - 1);

	uc = max(uc, 1);
	uc = min(uc, nDatau - 1);

	if (xi == float(nDatax))
		xc = nDatax;

	if (yi == float(nDatay))
		yc = nDatay;

	if (zi == float(nDataz))
		zc = nDataz;
	if (wi == float(nDataw))
		wc = nDataw;

	if (ui == float(nDatau))
		uc = nDatau;

	kx = xc + 1;
	ky = yc + 1;
	kz = zc + 1;
	kw = wc + 1;
	ku = uc + 1;
	for (s = 1; s <= 2; s++) {
		for (r = 1; r <= 2; r++) {
			for (q = 1; q <= 2; q++) {
				for (j = 1; j <= 2; j++) {
					for (i = 1; i <= 2; i++) {
						f += cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 1] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 2] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 3] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 4] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 5] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 6] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 7] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 8] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 9] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 10] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 11] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 12] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 13] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 14] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 15] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 16] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 17] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 18] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 19] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 20] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 21] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 22] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 23] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 24] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 25] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 26] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 27] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 28] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 29] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[32 * nIndex + 30] +
							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[32 * nIndex + 31];

						dfx += (-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex] +
							(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
								cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 1] +
								(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
									cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 2] +
									(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
										cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 3] +
										(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
											cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 4] +
											(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
												cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 5] +
												(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
													cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 6] +
													(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
														cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 7] +
														(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
															cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 8] +
															(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 9] +
																(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																	cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 10] +
																	(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																		cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 11] +
																		(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																			cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 12] +
																			(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																				cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 13] +
																				(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																					cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 14] +
																					(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																						cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 15] +
																						(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 16] +      //newone
																							(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																								cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 17] +
																								(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																									cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 18] +
																									(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																										cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 19] +
																										(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																											cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 20] +
																											(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																												cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 21] +
																												(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																													cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 22] +
																													(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																														cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 23] +
																														(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																															cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 24] +
																															(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 25] +
																																(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																	cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 26] +
																																	(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																		cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 27] +
																																		(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																																			cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 28] +
																																			(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																				cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 29] +
																																				(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																					cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[32 * nIndex + 30] +
																																					(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																						cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[32 * nIndex + 31];

						dfy += (-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex] +
							(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
								cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 1] +
								(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
									cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 2] +
									(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
										cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 3] +
										(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
											cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 4] +
											(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
												cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 5] +
												(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
													cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 6] +
													(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
														cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 7] +
														(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
															cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 8] +
															(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 9] +
																(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																	cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 10] +
																	(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																		cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 11] +
																		(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																			cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 12] +
																			(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																				cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 13] +
																				(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																					cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 14] +
																					(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																						cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 15] +
																						(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 16] +
																							(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																								cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 17] +
																								(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																									cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 18] +
																									(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																										cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 19] +
																										(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																											cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 20] +
																											(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																												cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 21] +
																												(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																													cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 22] +
																													(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																														cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 23] +
																														(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																															cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 24] +
																															(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 25] +
																																(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																	cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 26] +
																																	(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																		cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 27] +
																																		(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																																			cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 28] +
																																			(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																				cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 29] +
																																				(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																					cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[32 * nIndex + 30] +
																																					(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																						cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[32 * nIndex + 31];



						dfz += (-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex] +
							(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
								cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 1] +
								(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
									cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 2] +
									(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
										cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 3] +
										(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
											cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 4] +
											(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
												cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 5] +
												(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
													cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 6] +
													(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
														cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 7] +
														(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
															cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 8] +
															(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 9] +
																(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																	cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 10] +
																	(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																		cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 11] +
																		(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																			cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 12] +
																			(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																				cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 13] +
																				(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																					cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 14] +
																					(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																						cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 15] +
																						(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 16] +
																							(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																								cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 17] +
																								(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																									cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 18] +
																									(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																										cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 19] +
																										(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																											cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 20] +
																											(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																												cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 21] +
																												(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																													cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 22] +
																													(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																														cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 23] +
																														(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																															cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 24] +
																															(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 25] +
																																(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																	cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 26] +
																																	(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																		cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 271] +
																																		(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																																			cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 28] +
																																			(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																				cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 29] +
																																				(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																					cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[32 * nIndex + 30] +
																																					(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																						cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[32 * nIndex + 31];

						dfw += (-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
							cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex] +
							(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
								cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 1] +
								(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
									cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 2] +
									(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
										cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 3] +
										(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
											cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 4] +
											(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
												cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 5] +
												(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
													cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 6] +
													(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
														cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 7] +
														(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
															cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 8] +
															(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 9] +
																(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																	cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 10] +
																	(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																		cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 11] +
																		(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																			cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 12] +
																			(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																				cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 13] +
																				(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																					cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 14] +
																					(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																						cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 15] +
																						(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																							cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 16] +
																							(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																								cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 17] +
																								(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																									cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 18] +
																									(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																										cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 19] +
																										(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																											cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 20] +
																											(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																												cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 21] +
																												(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																													cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 22] +
																													(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																														cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 23] +
																														(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																															cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 24] +
																															(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 25] +
																																(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																	cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 26] +
																																	(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																		cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 27] +
																																		(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																																			cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 28] +
																																			(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																				cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 29] +
																																				(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																					cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[32 * nIndex + 30] +
																																					(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																						cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[32 * nIndex + 31];

						dfu += (-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
							cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex] +
							(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
								cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 1] +
								(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
									cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 2] +
									(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
										cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 3] +
										(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
											cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 4] +
											(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
												cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 5] +
												(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
													cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 6] +
													(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
														cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 7] +
														(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
															cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 8] +
															(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 9] +
																(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																	cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 10] +
																	(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																		cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 11] +
																		(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																			cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 12] +
																			(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																				cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 13] +
																				(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																					cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 14] +
																					(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																						cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 15] +
																						(-1.0 * cMat[(ku - s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																							cMat[(ku - s + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 16] +
																							(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																								cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 17] +
																								(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																									cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 18] +
																									(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																										cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 19] +
																										(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																											cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 20] +
																											(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																												cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 21] +
																												(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																													cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 22] +
																													(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																														cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 23] +
																														(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																															cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 24] +
																															(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 25] +
																																(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																	cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 26] +
																																	(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																		cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 27] +
																																		(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																																			cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 28] +
																																			(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																																				cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 29] +
																																				(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																																					cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfu[32 * nIndex + 30] +
																																					(-1.0 * cMat[(ku + s - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																																						cMat[(ku + s) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) * (nDataw + 3) + (kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfu[32 * nIndex + 31];




						nIndex = nIndex + 1;
					}
				}
			}
		}
	}
	//dudt[0]=-1.0f*theta[2]*dfy;
	//dudt[1]=-1.0f*theta[2]*dfx;
	//flipped in bSpline
	dudt[0] = -1.0f * theta[2] * dfy;
	dudt[1] = -1.0f * theta[2] * dfx;
	dudt[4] = theta[2] * dfz;
	dudt[5] = theta[2] * dfw;
	dudt[6] = theta[2] * dfu;
	dudt[2] = f;
	dudt[3] = 1.0f;
	*model = theta[3] + theta[2] * f;

	//return pd;
}

__device__ inline void kernel_luEvaluate5D(float* L, float* U, float* b, const int n, float* x) {
	//Ax = b -> LUx = b. Then y is defined to be Ux
	float y[8] = { 0 };
	int i = 0;
	int j = 0;
	// Forward solve Ly = b
	for (i = 0; i < n; i++)
	{
		y[i] = b[i];
		for (j = 0; j < i; j++)
		{
			y[i] -= L[j * n + i] * y[j];
		}
		y[i] /= L[i * n + i];
	}
	// Backward solve Ux = y
	for (i = n - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (j = i + 1; j < n; j++)
		{
			x[i] -= U[j * n + i] * x[j];
		}
		x[i] /= U[i * n + i];
	}

}


//**************************************************************************************************************

//4D
__device__ inline void kernel_computeDelta4D_bSpline(float x_delta, float y_delta, float z_delta, float w_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, float* delta_dwf) {
	float dx_delta, dy_delta, dz_delta, dw_delta;
	int nIndex = 0;
	int q, j, i, r;
	float Bw1, Bw2, dBw1, dBw2, Bz1, Bz2, dBz1, dBz2, Bx1, Bx2, dBx1, dBx2, By1, By2, dBy1, dBy2;

	dx_delta = x_delta - 1.0f;
	dy_delta = y_delta - 1.0f;
	dz_delta = z_delta - 1.0f;
	dw_delta = w_delta - 1.0f;

	for (r = 1; r <= 2; r++) {
		kernel_evalBSpline(w_delta + r - 1.0f, 3, &Bw1);
		kernel_evalBSpline(w_delta - r, 3, &Bw2);
		kernel_evalBSpline(dw_delta + r - 1.0f / 2.0f, 2, &dBw1);
		kernel_evalBSpline(dw_delta - r + 1.0f / 2.0f, 2, &dBw2);
		for (q = 1; q <= 2; q++) {
			kernel_evalBSpline(z_delta + q - 1.0f, 3, &Bz1);
			kernel_evalBSpline(z_delta - q, 3, &Bz2);
			kernel_evalBSpline(dz_delta + q - 1.0f / 2.0f, 2, &dBz1);
			kernel_evalBSpline(dz_delta - q + 1.0f / 2.0f, 2, &dBz2);

			for (j = 1; j <= 2; j++) {
				kernel_evalBSpline(x_delta + j - 1.0f, 3, &Bx1);
				kernel_evalBSpline(x_delta - j, 3, &Bx2);
				kernel_evalBSpline(dx_delta + j - 1.0f / 2.0f, 2, &dBx1);
				kernel_evalBSpline(dx_delta - j + 1.0f / 2.0f, 2, &dBx2);
				for (i = 1; i <= 2; i++) {
					kernel_evalBSpline(y_delta + i - 1.0f, 3, &By1);
					kernel_evalBSpline(y_delta - i, 3, &By2);
					kernel_evalBSpline(dy_delta + i - 1.0f / 2.0f, 2, &dBy1);
					kernel_evalBSpline(dy_delta - i + 1.0f / 2.0f, 2, &dBy2);

					delta_f[16 * nIndex] = By1 * Bx1 * Bz1 * Bw1;
					delta_f[16 * nIndex + 1] = By2 * Bx1 * Bz1 * Bw1;
					delta_f[16 * nIndex + 2] = By1 * Bx2 * Bz1 * Bw1;
					delta_f[16 * nIndex + 3] = By2 * Bx2 * Bz1 * Bw1;
					delta_f[16 * nIndex + 4] = By1 * Bx1 * Bz2 * Bw1;
					delta_f[16 * nIndex + 5] = By2 * Bx1 * Bz2 * Bw1;
					delta_f[16 * nIndex + 6] = By1 * Bx2 * Bz2 * Bw1;
					delta_f[16 * nIndex + 7] = By2 * Bx2 * Bz2 * Bw1;

					delta_f[16 * nIndex + 8] = By1 * Bx1 * Bz1 * Bw2;
					delta_f[16 * nIndex + 9] = By2 * Bx1 * Bz1 * Bw2;
					delta_f[16 * nIndex + 10] = By1 * Bx2 * Bz1 * Bw2;
					delta_f[16 * nIndex + 11] = By2 * Bx2 * Bz1 * Bw2;
					delta_f[16 * nIndex + 12] = By1 * Bx1 * Bz2 * Bw2;
					delta_f[16 * nIndex + 13] = By2 * Bx1 * Bz2 * Bw2;
					delta_f[16 * nIndex + 14] = By1 * Bx2 * Bz2 * Bw2;
					delta_f[16 * nIndex + 15] = By2 * Bx2 * Bz2 * Bw2;
					//dx
					delta_dxf[16 * nIndex] = dBy1 * Bx1 * Bz1 * Bw1;
					delta_dxf[16 * nIndex + 1] = dBy2 * Bx1 * Bz1 * Bw1;
					delta_dxf[16 * nIndex + 2] = dBy1 * Bx2 * Bz1 * Bw1;
					delta_dxf[16 * nIndex + 3] = dBy2 * Bx2 * Bz1 * Bw1;
					delta_dxf[16 * nIndex + 4] = dBy1 * Bx1 * Bz2 * Bw1;
					delta_dxf[16 * nIndex + 5] = dBy2 * Bx1 * Bz2 * Bw1;
					delta_dxf[16 * nIndex + 6] = dBy1 * Bx2 * Bz2 * Bw1;
					delta_dxf[16 * nIndex + 7] = dBy2 * Bx2 * Bz2 * Bw1;

					delta_dxf[16 * nIndex + 8] = dBy1 * Bx1 * Bz1 * Bw2;
					delta_dxf[16 * nIndex + 9] = dBy2 * Bx1 * Bz1 * Bw2;
					delta_dxf[16 * nIndex + 10] = dBy1 * Bx2 * Bz1 * Bw2;
					delta_dxf[16 * nIndex + 11] = dBy2 * Bx2 * Bz1 * Bw2;
					delta_dxf[16 * nIndex + 12] = dBy1 * Bx1 * Bz2 * Bw2;
					delta_dxf[16 * nIndex + 13] = dBy2 * Bx1 * Bz2 * Bw2;
					delta_dxf[16 * nIndex + 14] = dBy1 * Bx2 * Bz2 * Bw2;
					delta_dxf[16 * nIndex + 15] = dBy2 * Bx2 * Bz2 * Bw2;
					//dy
					delta_dyf[16 * nIndex] = By1 * dBx1 * Bz1 * Bw1;
					delta_dyf[16 * nIndex + 1] = By2 * dBx1 * Bz1 * Bw1;
					delta_dyf[16 * nIndex + 2] = By1 * dBx2 * Bz1 * Bw1;
					delta_dyf[16 * nIndex + 3] = By2 * dBx2 * Bz1 * Bw1;
					delta_dyf[16 * nIndex + 4] = By1 * dBx1 * Bz2 * Bw1;
					delta_dyf[16 * nIndex + 5] = By2 * dBx1 * Bz2 * Bw1;
					delta_dyf[16 * nIndex + 6] = By1 * dBx2 * Bz2 * Bw1;
					delta_dyf[16 * nIndex + 7] = By2 * dBx2 * Bz2 * Bw1;

					delta_dyf[16 * nIndex + 8] = By1 * dBx1 * Bz1 * Bw2;
					delta_dyf[16 * nIndex + 9] = By2 * dBx1 * Bz1 * Bw2;
					delta_dyf[16 * nIndex + 10] = By1 * dBx2 * Bz1 * Bw2;
					delta_dyf[16 * nIndex + 11] = By2 * dBx2 * Bz1 * Bw2;
					delta_dyf[16 * nIndex + 12] = By1 * dBx1 * Bz2 * Bw2;
					delta_dyf[16 * nIndex + 13] = By2 * dBx1 * Bz2 * Bw2;
					delta_dyf[16 * nIndex + 14] = By1 * dBx2 * Bz2 * Bw2;
					delta_dyf[16 * nIndex + 15] = By2 * dBx2 * Bz2 * Bw2;
					//dz
					delta_dzf[16 * nIndex] = By1 * Bx1 * dBz1 * Bw1;
					delta_dzf[16 * nIndex + 1] = By2 * Bx1 * dBz1 * Bw1;
					delta_dzf[16 * nIndex + 2] = By1 * Bx2 * dBz1 * Bw1;
					delta_dzf[16 * nIndex + 3] = By2 * Bx2 * dBz1 * Bw1;
					delta_dzf[16 * nIndex + 4] = By1 * Bx1 * dBz2 * Bw1;
					delta_dzf[16 * nIndex + 5] = By2 * Bx1 * dBz2 * Bw1;
					delta_dzf[16 * nIndex + 6] = By1 * Bx2 * dBz2 * Bw1;
					delta_dzf[16 * nIndex + 7] = By2 * Bx2 * dBz2 * Bw1;

					delta_dzf[16 * nIndex + 8] = By1 * Bx1 * dBz1 * Bw2;
					delta_dzf[16 * nIndex + 9] = By2 * Bx1 * dBz1 * Bw2;
					delta_dzf[16 * nIndex + 10] = By1 * Bx2 * dBz1 * Bw2;
					delta_dzf[16 * nIndex + 11] = By2 * Bx2 * dBz1 * Bw2;
					delta_dzf[16 * nIndex + 12] = By1 * Bx1 * dBz2 * Bw2;
					delta_dzf[16 * nIndex + 13] = By2 * Bx1 * dBz2 * Bw2;
					delta_dzf[16 * nIndex + 14] = By1 * Bx2 * dBz2 * Bw2;
					delta_dzf[16 * nIndex + 15] = By2 * Bx2 * dBz2 * Bw2;
					//dw
					delta_dwf[16 * nIndex] = By1 * Bx1 * Bz1 * dBw1;
					delta_dwf[16 * nIndex + 1] = By2 * Bx1 * Bz1 * dBw1;
					delta_dwf[16 * nIndex + 2] = By1 * Bx2 * Bz1 * dBw1;
					delta_dwf[16 * nIndex + 3] = By2 * Bx2 * Bz1 * dBw1;
					delta_dwf[16 * nIndex + 4] = By1 * Bx1 * Bz2 * dBw1;
					delta_dwf[16 * nIndex + 5] = By2 * Bx1 * Bz2 * dBw1;
					delta_dwf[16 * nIndex + 6] = By1 * Bx2 * Bz2 * dBw1;
					delta_dwf[16 * nIndex + 7] = By2 * Bx2 * Bz2 * dBw1;

					delta_dwf[16 * nIndex + 8] = By1 * Bx1 * Bz1 * dBw2;
					delta_dwf[16 * nIndex + 9] = By2 * Bx1 * Bz1 * dBw2;
					delta_dwf[16 * nIndex + 10] = By1 * Bx2 * Bz1 * dBw2;
					delta_dwf[16 * nIndex + 11] = By2 * Bx2 * Bz1 * dBw2;
					delta_dwf[16 * nIndex + 12] = By1 * Bx1 * Bz2 * dBw2;
					delta_dwf[16 * nIndex + 13] = By2 * Bx1 * Bz2 * dBw2;
					delta_dwf[16 * nIndex + 14] = By1 * Bx2 * Bz2 * dBw2;
					delta_dwf[16 * nIndex + 15] = By2 * Bx2 * Bz2 * dBw2;
					/* delta_dxf[8 * nIndex] = dBy1 * Bx1 * Bz1;
					 delta_dxf[8 * nIndex + 1] = dBy2 * Bx1 * Bz1;
					 delta_dxf[8 * nIndex + 2] = dBy1 * Bx2 * Bz1;
					 delta_dxf[8 * nIndex + 3] = dBy2 * Bx2 * Bz1;
					 delta_dxf[8 * nIndex + 4] = dBy1 * Bx1 * Bz2;
					 delta_dxf[8 * nIndex + 5] = dBy2 * Bx1 * Bz2;
					 delta_dxf[8 * nIndex + 6] = dBy1 * Bx2 * Bz2;
					 delta_dxf[8 * nIndex + 7] = dBy2 * Bx2 * Bz2;

					 delta_dyf[8 * nIndex] = By1 * dBx1 * Bz1;
					 delta_dyf[8 * nIndex + 1] = By2 * dBx1 * Bz1;
					 delta_dyf[8 * nIndex + 2] = By1 * dBx2 * Bz1;
					 delta_dyf[8 * nIndex + 3] = By2 * dBx2 * Bz1;
					 delta_dyf[8 * nIndex + 4] = By1 * dBx1 * Bz2;
					 delta_dyf[8 * nIndex + 5] = By2 * dBx1 * Bz2;
					 delta_dyf[8 * nIndex + 6] = By1 * dBx2 * Bz2;
					 delta_dyf[8 * nIndex + 7] = By2 * dBx2 * Bz2;

					 delta_dzf[8 * nIndex] = By1 * Bx1 * dBz1;
					 delta_dzf[8 * nIndex + 1] = By2 * Bx1 * dBz1;
					 delta_dzf[8 * nIndex + 2] = By1 * Bx2 * dBz1;
					 delta_dzf[8 * nIndex + 3] = By2 * Bx2 * dBz1;
					 delta_dzf[8 * nIndex + 4] = By1 * Bx1 * dBz2;
					 delta_dzf[8 * nIndex + 5] = By2 * Bx1 * dBz2;
					 delta_dzf[8 * nIndex + 6] = By1 * Bx2 * dBz2;
					 delta_dzf[8 * nIndex + 7] = By2 * Bx2 * dBz2;*/

					nIndex = nIndex + 1;
				}
			}
		}
	}
}

//**************************************************************************************************************

__device__ inline void kernel_Derivative_bSpline14D(float xi, float yi, float zi, float wi, int nDatax, int nDatay, int nDataz, int nDataw, float* delta_f, float* delta_dfx, float* delta_dfy, float* delta_dfz, float* delta_dfw, const float* cMat, float* theta, float* dudt, float* model) {
	int xc, yc, zc, wc, kx, ky, kz, kw, q, j, i, r;
	int nIndex = 0.0f;
	float f = 0.0f, dfx = 0.0f, dfy = 0.0f, dfz = 0.0f, dfw = 0.0f;;

	xc = floor(xi);
	yc = floor(yi);
	zc = floor(zi);
	wc = floor(wi);

	xc = max(xc, 1);
	xc = min(xc, nDatax - 1);

	yc = max(yc, 1);
	yc = min(yc, nDatay - 1);

	zc = max(zc, 1);
	zc = min(zc, nDataz - 1);

	wc = max(wc, 1);
	wc = min(wc, nDataw - 1);

	if (xi == float(nDatax))
		xc = nDatax;

	if (yi == float(nDatay))
		yc = nDatay;

	if (zi == float(nDataz))
		zc = nDataz;
	if (wi == float(nDataw))
		wc = nDataw;

	kx = xc + 1;
	ky = yc + 1;
	kz = zc + 1;
	kw = wc + 1;
	for (r = 1; r <= 2; r++) {
		for (q = 1; q <= 2; q++) {
			for (j = 1; j <= 2; j++) {
				for (i = 1; i <= 2; i++) {
					f += cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 1] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 2] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 3] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 4] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 5] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 6] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 7] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 8] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 9] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 10] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 11] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 12] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 13] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[16 * nIndex + 14] +
						cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[16 * nIndex + 15];

					dfx += (-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex] +
						(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
							cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 1] +
							(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
								cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 2] +
								(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
									cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 3] +
									(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
										cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 4] +
										(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
											cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 5] +
											(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
												cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 6] +
												(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
													cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 7] +
													(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
														cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 8] +
														(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
															cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 9] +
															(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 10] +
																(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																	cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 11] +
																	(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																		cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 12] +
																		(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																			cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 13] +
																			(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																				cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[16 * nIndex + 14] +
																				(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																					cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[16 * nIndex + 15];

					dfy += (-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex] +
						(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
							cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 1] +
							(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
								cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex + 2] +
								(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
									cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 3] +
									(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
										cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex + 4] +
										(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
											cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 5] +
											(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
												cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex + 6] +
												(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
													cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 7] +
													(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
														cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex] +
														(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
															cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 1] +
															(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex + 2] +
																(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																	cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 3] +
																	(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																		cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex + 4] +
																		(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																			cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 5] +
																			(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																				cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[16 * nIndex + 6] +
																				(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																					cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[16 * nIndex + 7];


					dfz += (-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
						cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex] +
						(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
							cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 1] +
							(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
								cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 2] +
								(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
									cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 3] +
									(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
										cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 4] +
										(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
											cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 5] +
											(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
												cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 6] +
												(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
													cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 7] +
													(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
														cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 8] +
														(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
															cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 9] +
															(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 10] +
																(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																	cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 11] +
																	(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																		cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 12] +
																		(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																			cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 13] +
																			(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																				cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[16 * nIndex + 14] +
																				(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																					cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[16 * nIndex + 15];

					dfw += (-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
						cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex] +
						(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
							cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 1] +
							(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
								cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 2] +
								(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
									cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 3] +
									(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
										cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 4] +
										(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
											cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 5] +
											(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
												cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 6] +
												(-1.0 * cMat[(kw - r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
													cMat[(kw - r + 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 7] +
													(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
														cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 8] +
														(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
															cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 9] +
															(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 10] +
																(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																	cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 11] +
																	(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] +
																		cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 12] +
																		(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] +
																			cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 13] +
																			(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] +
																				cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfw[16 * nIndex + 14] +
																				(-1.0 * cMat[(kw + r - 1) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] +
																					cMat[(kw + r) * (nDatax + 3) * (nDatay + 3) * (nDataz + 3) + (kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfw[16 * nIndex + 15];

					nIndex = nIndex + 1;
				}
			}
		}
	}

	//dudt[0]=-1.0f*theta[2]*dfy;
	//dudt[1]=-1.0f*theta[2]*dfx;
	//flipped in bSpline
	dudt[0] = -1.0f * theta[2] * dfy;
	dudt[1] = -1.0f * theta[2] * dfx;
	dudt[4] = theta[2] * dfz;
	dudt[5] = theta[2] * dfw;
	dudt[2] = f;
	dudt[3] = 1.0f;
	*model = theta[3] + theta[2] * f;

	//return pd;
}


//**************************************************************************************************************



__device__ inline void kernel_computeDelta3D(float x_delta, float y_delta, float z_delta, float *delta_f, float *delta_dxf, float *delta_dyf, float *delta_dzf) {
    
	int i,j,k;
	float cx,cy,cz;

	cz = 1.0;
	for(i=0;i<4;i++){
		cy = 1.0;
		for(j=0;j<4;j++){
			cx = 1.0;
			for(k=0;k<4;k++){
				delta_f[i*16+j*4+k] = cz * cy * cx;
				if(k<3){
					delta_dxf[i*16+j*4+k+1] = ((float)k+1) * cz * cy * cx;
				}
				
				if(j<3){
					delta_dyf[i*16+(j+1)*4+k] = ((float)j+1) * cz * cy * cx;
				}
				
				if(i<3){
					delta_dzf[(i+1)*16+j*4+k] = ((float)i+1) * cz * cy * cx;
				}
				
				cx = cx * x_delta;
			}
			cy = cy * y_delta;
		}
		cz= cz * z_delta;
	}
}


//***********************************************************************************************************
__device__ inline int kernel_cholesky(float *A,int n, float *L, float*U) {
	int info = 0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < (i+1); j++) {
			float s = 0;
			for (int k = 0; k < j; k++)
				s += U[i * n + k] * U[j * n + k];

			if (i==j){
				if (A[i*n+i]-s>=0){
					U[i * n + j] = sqrt(A[i * n + i] - s);
					L[j*n+i]=U[i * n + j];
				}
				else{
					info =1;
					return info;
				}
			}
			else{
				U[i * n + j] = (1.0 / U[j * n + j] * (A[i * n + j] - s));
				L[j*n+i]=U[i * n + j];
			}

		}
	return info;
}
//******************************************************************************************************
__device__ inline void kernel_luEvaluate(float *L,float *U, float *b, const int n, float *x) {
	//Ax = b -> LUx = b. Then y is defined to be Ux
	//for sigmaxy, we have 6 parameters
	float y[6] = {0};
	int i = 0;
	int j = 0;
	// Forward solve Ly = b
	for (i = 0; i < n; i++)
	{
		y[i] = b[i];
		for (j = 0; j < i; j++)
		{
			y[i] -= L[j*n+i] * y[j];
		}
		y[i] /= L[i*n+i];
	}
	// Backward solve Ux = y
	for (i = n - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (j = i + 1; j < n; j++)
		{
			x[i] -= U[j*n+i] * x[j];
		}
		x[i] /= U[i*n+i];
	}

}


//**************************************************************************************************************

__device__ inline void kernel_DerivativeSpline(int xc, int yc, int zc, int xsize, int ysize, int zsize, float *delta_f, float *delta_dxf, float *delta_dyf, float *delta_dzf,const float *coeff,float *theta, float*dudt,float*model) {
	int i;
	float temp =0;
	//float dudt_temp[NV_PSP] = {0};//,temp;
	memset(dudt,0,NV_PSP*sizeof(float));
	//for (i=0;i<NV_PSP;i++) dudt[i]=0;
	
	xc = max(xc,0);
	xc = min(xc,xsize-1);

	yc = max(yc,0);
	yc = min(yc,ysize-1);

	zc = max(zc,0);
	zc = min(zc,zsize-1);
	
	

	for (i=0;i<64;i++){		
		temp+=delta_f[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc];
		dudt[0]+=delta_dxf[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc];
		dudt[1]+=delta_dyf[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc];
		dudt[4]+=delta_dzf[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc];
				//temp = tex1Dfetch(tex_test, i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc);
				//pd+=delta_f[i]*temp;
	}
	dudt[0]*=-1.0f*theta[2];
	dudt[1]*=-1.0f*theta[2];
	dudt[4]*=theta[2];
	dudt[2]=temp;
	dudt[3]=1.0f;
	*model = theta[3]+theta[2]*temp;
	
	//return pd;
}

//**************************************************************************************************************

__device__ inline void kernel_evalBSpline(float xi, int deg, float* bi) {
	float x2, x3;



	//switch(deg){
	//case 2:
	//	if (xi>=1.0f/2.0f&&xi<3.0f/2.0f){
	//		*bi = 9.0f/8.0f-3.0f/2.0f*xi+1.0f/2.0f*x2;
	//	}
	//	if (xi>=-1.0f/2.0f&&xi<1.0f/2.0f){
	//		*bi = 3.0f/4.0f-x2;
	//	}
	//	if (xi>=-3.0f/2.0f&&xi<-1.0f/2.0f){
	//		*bi = 9.0f/8.0f+3.0f/2.0f*xi+1.0f/2.0f*x2;
	//	}
	//case 3:
	//	if (xi>=1.0f&&xi<2.0f){
	//		*bi = 4.0f/3.0f-2.0f*xi+x2-1.0f/6.0f*x3;
	//	}
	//	if (xi>=0.0f&&xi<1.0f){
	//		*bi = 2.0f/3.0f-x2+1.0f/2.0f*x3;
	//	}
	//	if (xi>=-1.0f&&xi<0.0f){
	//		*bi = 2.0f/3.0f-x2-1.0f/2.0f*x3;
	//	}
	//	if (xi>=-2.0f&&xi<-1.0f){
	//		*bi = 4.0f/3.0f+2.0f*xi+x2+1.0f/6.0f*x3;
	//	}
	//}
	* bi = 0;
	switch (deg) {
	case 2:
		x2 = xi * xi;
		if (xi >= 0.5f && xi < 1.5f) {
			*bi = 1.125f - 1.5f * xi + 0.5f * x2;
		}
		else if (xi >= -0.5f && xi < 0.5f) {
			*bi = 0.75f - x2;
		}
		else if (xi >= -1.5f && xi < -0.5f) {
			*bi = 1.125f + 1.5 * xi + 0.5f * x2;
		}

		break;
	case 3:
		x2 = xi * xi;
		x3 = x2 * xi;
		if (xi >= 1.0f && xi < 2.0f) {
			*bi = 4.0f / 3.0f - 2.0f * xi + x2 - 1.0f / 6.0f * x3;
		}
		else if (xi >= 0.0f && xi < 1.0f) {
			*bi = 2.0f / 3.0f - x2 + 0.5f * x3;
		}
		else if (xi >= -1.0f && xi < 0.0f) {
			*bi = 2.0f / 3.0f - x2 - 0.5f * x3;
		}
		else if (xi >= -2.0f && xi < -1.0f) {
			*bi = 4.0f / 3.0f + 2.0f * xi + x2 + 1.0f / 6.0f * x3;
		}

	}
}


__device__ inline void kernel_computeDelta3D_bSpline(float x_delta, float y_delta, float z_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf) {
	float dx_delta, dy_delta, dz_delta;
	int nIndex = 0;
	int q, j, i;
	float Bz1, Bz2, dBz1, dBz2, Bx1, Bx2, dBx1, dBx2, By1, By2, dBy1, dBy2;

	dx_delta = x_delta - 1.0f;
	dy_delta = y_delta - 1.0f;
	dz_delta = z_delta - 1.0f;

	for (q = 1; q <= 2; q++) {
		kernel_evalBSpline(z_delta + q - 1.0f, 3, &Bz1);
		kernel_evalBSpline(z_delta - q, 3, &Bz2);
		kernel_evalBSpline(dz_delta + q - 1.0f / 2.0f, 2, &dBz1);
		kernel_evalBSpline(dz_delta - q + 1.0f / 2.0f, 2, &dBz2);

		for (j = 1; j <= 2; j++) {
			kernel_evalBSpline(x_delta + j - 1.0f, 3, &Bx1);
			kernel_evalBSpline(x_delta - j, 3, &Bx2);
			kernel_evalBSpline(dx_delta + j - 1.0f / 2.0f, 2, &dBx1);
			kernel_evalBSpline(dx_delta - j + 1.0f / 2.0f, 2, &dBx2);
			for (i = 1; i <= 2; i++) {
				kernel_evalBSpline(y_delta + i - 1.0f, 3, &By1);
				kernel_evalBSpline(y_delta - i, 3, &By2);
				kernel_evalBSpline(dy_delta + i - 1.0f / 2.0f, 2, &dBy1);
				kernel_evalBSpline(dy_delta - i + 1.0f / 2.0f, 2, &dBy2);

				delta_f[8 * nIndex] = By1 * Bx1 * Bz1;
				delta_f[8 * nIndex + 1] = By2 * Bx1 * Bz1;
				delta_f[8 * nIndex + 2] = By1 * Bx2 * Bz1;
				delta_f[8 * nIndex + 3] = By2 * Bx2 * Bz1;
				delta_f[8 * nIndex + 4] = By1 * Bx1 * Bz2;
				delta_f[8 * nIndex + 5] = By2 * Bx1 * Bz2;
				delta_f[8 * nIndex + 6] = By1 * Bx2 * Bz2;
				delta_f[8 * nIndex + 7] = By2 * Bx2 * Bz2;

				delta_dxf[8 * nIndex] = dBy1 * Bx1 * Bz1;
				delta_dxf[8 * nIndex + 1] = dBy2 * Bx1 * Bz1;
				delta_dxf[8 * nIndex + 2] = dBy1 * Bx2 * Bz1;
				delta_dxf[8 * nIndex + 3] = dBy2 * Bx2 * Bz1;
				delta_dxf[8 * nIndex + 4] = dBy1 * Bx1 * Bz2;
				delta_dxf[8 * nIndex + 5] = dBy2 * Bx1 * Bz2;
				delta_dxf[8 * nIndex + 6] = dBy1 * Bx2 * Bz2;
				delta_dxf[8 * nIndex + 7] = dBy2 * Bx2 * Bz2;

				delta_dyf[8 * nIndex] = By1 * dBx1 * Bz1;
				delta_dyf[8 * nIndex + 1] = By2 * dBx1 * Bz1;
				delta_dyf[8 * nIndex + 2] = By1 * dBx2 * Bz1;
				delta_dyf[8 * nIndex + 3] = By2 * dBx2 * Bz1;
				delta_dyf[8 * nIndex + 4] = By1 * dBx1 * Bz2;
				delta_dyf[8 * nIndex + 5] = By2 * dBx1 * Bz2;
				delta_dyf[8 * nIndex + 6] = By1 * dBx2 * Bz2;
				delta_dyf[8 * nIndex + 7] = By2 * dBx2 * Bz2;

				delta_dzf[8 * nIndex] = By1 * Bx1 * dBz1;
				delta_dzf[8 * nIndex + 1] = By2 * Bx1 * dBz1;
				delta_dzf[8 * nIndex + 2] = By1 * Bx2 * dBz1;
				delta_dzf[8 * nIndex + 3] = By2 * Bx2 * dBz1;
				delta_dzf[8 * nIndex + 4] = By1 * Bx1 * dBz2;
				delta_dzf[8 * nIndex + 5] = By2 * Bx1 * dBz2;
				delta_dzf[8 * nIndex + 6] = By1 * Bx2 * dBz2;
				delta_dzf[8 * nIndex + 7] = By2 * Bx2 * dBz2;

				nIndex = nIndex + 1;
			}
		}
	}

}

__device__ inline void kernel_Derivative_bSpline1(float xi, float yi, float zi, int nDatax, int nDatay, int nDataz, float* delta_f, float* delta_dfx, float* delta_dfy, float* delta_dfz, const float* cMat, float* theta, float* dudt, float* model) {
	int xc, yc, zc, kx, ky, kz, q, j, i;
	int nIndex = 0.0f;
	float f = 0.0f, dfx = 0.0f, dfy = 0.0f, dfz = 0.0f;

	xc = floor(xi);
	yc = floor(yi);
	zc = floor(zi);

	xc = max(xc, 1);
	xc = min(xc, nDatax - 1);

	yc = max(yc, 1);
	yc = min(yc, nDatay - 1);

	zc = max(zc, 1);
	zc = min(zc, nDataz - 1);

	if (xi == float(nDatax))
		xc = nDatax;

	if (yi == float(nDatay))
		yc = nDatay;

	if (zi == float(nDataz))
		zc = nDataz;

	kx = xc + 1;
	ky = yc + 1;
	kz = zc + 1;

	for (q = 1; q <= 2; q++) {
		for (j = 1; j <= 2; j++) {
			for (i = 1; i <= 2; i++) {
				f += cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[8 * nIndex] +
					cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[8 * nIndex + 1] +
					cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[8 * nIndex + 2] +
					cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[8 * nIndex + 3] +
					cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] * delta_f[8 * nIndex + 4] +
					cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] * delta_f[8 * nIndex + 5] +
					cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] * delta_f[8 * nIndex + 6] +
					cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] * delta_f[8 * nIndex + 7];

				dfx += (-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[8 * nIndex] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[8 * nIndex + 1] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[8 * nIndex + 2] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[8 * nIndex + 3] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[8 * nIndex + 4] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i)]) * delta_dfx[8 * nIndex + 5] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i + 1)]) * delta_dfx[8 * nIndex + 6] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i)]) * delta_dfx[8 * nIndex + 7];

				dfy += (-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[8 * nIndex] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[8 * nIndex + 1] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[8 * nIndex + 2] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[8 * nIndex + 3] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky - i)]) * delta_dfy[8 * nIndex + 4] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j + 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[8 * nIndex + 5] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky - i)]) * delta_dfy[8 * nIndex + 6] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfy[8 * nIndex + 7];


				dfz += (-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] + cMat[(kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[8 * nIndex] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[8 * nIndex + 1] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] + cMat[(kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[8 * nIndex + 2] +
					(-1.0 * cMat[(kz - q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz - q + 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[8 * nIndex + 3] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)] + cMat[(kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky - i)]) * delta_dfz[8 * nIndex + 4] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz + q) * (nDatax + 3) * (nDatay + 3) + (kx - j) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[8 * nIndex + 5] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)] + cMat[(kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky - i)]) * delta_dfz[8 * nIndex + 6] +
					(-1.0 * cMat[(kz + q - 1) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)] + cMat[(kz + q) * (nDatax + 3) * (nDatay + 3) + (kx + j - 1) * (nDatax + 3) + (ky + i - 1)]) * delta_dfz[8 * nIndex + 7];

				nIndex = nIndex + 1;
			}
		}
	}


	//dudt[0]=-1.0f*theta[2]*dfy;
	//dudt[1]=-1.0f*theta[2]*dfx;
	//flipped in bSpline
	dudt[0] = -1.0f * theta[2] * dfy;
	dudt[1] = -1.0f * theta[2] * dfx;
	dudt[4] = theta[2] * dfz;
	dudt[2] = f;
	dudt[3] = 1.0f;
	*model = theta[3] + theta[2] * f;

	//return pd;
}

