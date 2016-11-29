#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE_WIDTH 36
#define TILE_HEIGHT 36

__device__ int calc_mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__device__ int mandelbrot_calc(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int row, int col,
    int maxIterations)
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
	float x = x0 + col * dx;
	float y = y0 + row * dy;
	int val = calc_mandel(x, y, maxIterations);

	return val;
}

__global__ void mandelbrot_kernel(float x0, float y0, float x1, float y1, int width, int height, int maxIterations, int* output) {
	int col = blockIdx.x*blockDim.x + threadIdx.x;	
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < height && col < width) {
		output[row*width + col] = mandelbrot_calc(x0, y0, x1, y1, width, height, row, col, maxIterations);
	}
}

void mandelbrotGpu(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations,
    int output[]) {

	int* d_output;
	float millisec = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	cudaMalloc(&d_output, width*height*sizeof(int));

	int nTilesX = width / TILE_WIDTH + (width % TILE_WIDTH == 0) ? 0 : 1;
	int nTilesY = height / TILE_HEIGHT + (height % TILE_HEIGHT == 0) ? 0 : 1;
	dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
	dim3 blocksPerGrid(nTilesX, nTilesY);
	mandelbrot_kernel<<<blocksPerGrid, threadsPerBlock>>>(x0, y0, x1, y1, width, height, maxIterations, d_output);
	
	cudaMemcpy(output, d_output, width*height*sizeof(int), cudaMemcpyDeviceToHost);		
	
	cudaFree(d_output);		

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisec, start, stop);
	printf("\ncuda time = %f\n", millisec);
}
