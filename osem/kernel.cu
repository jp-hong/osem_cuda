

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "parameters.h"
#include "util.h"
#include "cuda_osem.cuh"


void loadImage(float *d_image1D, int nBytes, int len, char *fileName);

void loadImagePinnedMem(float *h_data, float *d_data, int nBytes, int len, char *fileName);

void saveImage(float *d_data1D, int len, int nBytes, char *fileName);

inline int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

inline float getAngle(int view)
{
	return S0 + view * SLEN / 360.0f;
}

inline void startGPUTimer(cudaEvent_t start)
{
	HANDLE_ERROR(cudaEventRecord(start));
}

inline void stopGPUTimer(cudaEvent_t stop)
{
	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));
}

inline float getElapsedTime(cudaEvent_t start, cudaEvent_t stop)
{
	float ms;
	HANDLE_ERROR(cudaEventElapsedTime(&ms, start, stop));
	return ms;
}

int main()
{
	//GPU timer
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	float ms;
	
	//1D block for general use
	dim3 block(512);
	dim3 grid(0);

	//2D block for detector array (forward projection)
	dim3 block2d(32, 16);
	dim3 grid2d(iDivUp(NU, block2d.x), iDivUp(NV, block2d.y));

	//3D block for image volume (back projection)
	dim3 block3d(16, 8, 8);
	dim3 grid3d(iDivUp(NX, block3d.x), iDivUp(NY, block3d.y), iDivUp(NZ, block3d.z));

	//All memory allocation
	char *fileName = (char *)malloc(sizeof(char) * 512);

	float *h_data, *data, *proj, *image, *diffImage, *normImage;

	HANDLE_ERROR(cudaMalloc((void **)&data, PROJ_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&proj, PROJ_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&image, IMAGE_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&diffImage, IMAGE_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&normImage, IMAGE_BYTES));
	HANDLE_ERROR(cudaMallocHost((void **)&h_data, PROJ_BYTES));

	//CUDA array for 3D texture
	const cudaExtent volumeSize = make_cudaExtent(NX, NY, NZ);
	createCudaArray(volumeSize);

	float angle, cosbeta, sinbeta;
	int deltaBeta = 1;

	float centernBins = (NU - 1.0) / 2.0;
	float ScaleFactor = R / D;
	float deltaS = DU * ScaleFactor;
	float centernZZ = (NV - 1.0) / 2.0;
	float deltaZZ0 = DV;
	float deltaZZ = deltaZZ0 * ScaleFactor;

	float lamda = LAMDA;

	//reconstruction
	printf("\n --- STARTING RECONSTRUCTION WITH OSEM ---\n\n");

	getGridDim(&grid, block, IMAGE_LEN);
	setArrVal <<<grid, block>>> (image, IMAGE_LEN, 1);
	HANDLE_ERROR(cudaMemset(diffImage, 0, IMAGE_BYTES));

	for (int i = 0; i < ITER; i++)
	{
		printf("Iteration : %03d ... ", i + 1);
		startGPUTimer(start);

		bind3DTexture1(image, volumeSize);
		HANDLE_ERROR(cudaMemset(normImage, 0, IMAGE_BYTES));

		for (int j = i % N_SUBSET; j < NS; j += N_SUBSET)
		{
			angle = getAngle(j);
			cosbeta = cosf(angle);
			sinbeta = sinf(angle);

			generateFileName(fileName, INPUT_DIR, "", j, ".dat");
			loadImagePinnedMem(h_data, data, PROJ_BYTES, PROJ_LEN, fileName);

			forwardProjectionTexKernel <<<grid2d, block2d>>> (proj, sinbeta, cosbeta,
				R, D, NU, NV, DU, DV, U0, V0, DX, DY, DZ, X0, Y0, Z0, NX, NY, NZ, XLEN, YLEN, ZLEN);

			getGridDim(&grid, block, PROJ_LEN);
			divisionKernel <<<grid, block>>> (data, proj, PROJ_LEN);

			backProjectionOSEMKernel <<<grid3d, block3d>>> (diffImage, data, R, cosbeta,
				sinbeta, deltaZZ, centernZZ, deltaS, centernBins, NV, NU, deltaBeta, normImage,
				NX, NY, NZ, DX, DY, DZ);

			getGridDim(&grid, block, IMAGE_LEN);
			nanAndInfCheck <<<grid, block>>> (diffImage, IMAGE_LEN);
		}

		OSEMUpdateKernel <<<grid, block>>> (image, diffImage, normImage, IMAGE_LEN, lamda);
		makePositive <<<grid, block>>> (image, IMAGE_LEN);
		HANDLE_ERROR(cudaMemset(diffImage, 0, IMAGE_BYTES));

		lamda *= REG_FAC;

		stopGPUTimer(stop);
		ms = getElapsedTime(start, stop);
		printf("Elapsed time : %.3f ms\n", ms);

		if (i == 0 || (i + 1) % SAVE_INTERVAL == 0)
		{
			generateFileName(fileName, OUTPUT_DIR, SAVE_FILE_NAME, i + 1, ".dat");
			saveImage(image, IMAGE_LEN, IMAGE_BYTES, fileName);
		}
	}

	printf("\n --- RECONSTRUCTION FINISHED AFTER %d ITERATIONS ---\n\n", ITER);

	//Free all allocated memory
	free3DTexture();
	free(fileName);
	HANDLE_ERROR(cudaFree(data));
	HANDLE_ERROR(cudaFree(proj));
	HANDLE_ERROR(cudaFree(image));
	HANDLE_ERROR(cudaFree(diffImage));
	HANDLE_ERROR(cudaFree(normImage));
	HANDLE_ERROR(cudaFreeHost(h_data));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	//Reset device state
	HANDLE_ERROR(cudaDeviceReset());

	return 0;
}

void loadImage(float *d_image1D, int nBytes, int len, char *fileName)
{
	float *h_image1D = (float*)malloc(nBytes);
	readArrayFromFile(h_image1D, len, fileName, sizeof(float));
	HANDLE_ERROR(cudaMemcpy(d_image1D, h_image1D, nBytes, cudaMemcpyHostToDevice));
	free(h_image1D);
}

void loadImagePinnedMem(float *h_data, float *d_data, int nBytes, int len, char *fileName)
{
	readArrayFromFile(h_data, PROJ_LEN, fileName, sizeof(float));
	HANDLE_ERROR(cudaMemcpy(d_data, h_data, PROJ_BYTES, cudaMemcpyHostToDevice));
}

void saveImage(float *d_data1D, int len, int nBytes, char *fileName)
{
	float *h_data1D = (float*)malloc(nBytes);
	HANDLE_ERROR(cudaMemcpy(h_data1D, d_data1D, nBytes, cudaMemcpyDeviceToHost));
	writeArrayToFile(h_data1D, len, fileName, sizeof(float));
	free(h_data1D);
	printf("Saved file \"%s\"\n", fileName);
}