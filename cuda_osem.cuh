

#ifndef _CUDA_OSEM
#define _CUDA_OSEM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "util.h"

void setTextureFilterMode(bool bLinearFilter);

void createCudaArray(cudaExtent volumeSize);

void bind3DTexture1(const float *d_volume, cudaExtent volumeSize);

void bind3DTexture2(const float *d_volume, cudaExtent volumeSize);

void free3DTexture();

__global__ void rotationKernel(float *img, int nx, int ny, int nz, float x0, float y0, float z0,
	float dx, float dy, float dz, float xlen, float ylen, float zlen, float delta_angle);

__global__ void forwardProjectionTexKernel(float *worksino, float sinval, float cosval, float r,
	float s, int nu, int nv, float du, float dv, float u0, float v0, float dx, float dy, float dz,
	float x0, float y0, float z0, int nx, int ny, int nz, float xlen, float ylen, float zlen);

__global__ void backProjectionOSEMKernel(float *d_image, float *d_cons3, float F, float cosbeta, float sinbeta, float deltaZZ,
	float centernZZ, float deltaS, float centerBins, int nZZ, int nBins, int deltaBeta, float *d_norm, int nx, int ny, int nz, float dx, float dy, float dz);

__global__ void correctiveImageKernel(float *data, float *proj, float *corImg, int len);

__global__ void makePositive(float *image, int len);

__global__ void nanAndInfCheck(float *f, int len);

__global__ void divisionKernel(float *a, float *b, int len);

__global__ void additionKernel(float *a, float *b, int len);

__global__ void imageUpdatekernel(float *a, float *b, int len, float lamda);

__global__ void OSEMUpdateKernel(float *image, float *diffImage, float *normImage, int len, float lamda);

__global__ void setArrVal(float *arr, int len, float val);

__global__ void dataFlipV(float *data, int nu, int nv);

__global__ void dataFlipU(float *data, int nu, int nv);

#endif