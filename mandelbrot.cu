#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOLERANCE 0.00001
#define TRUE 1
#define FALSE 0

long usecs();
void initialize(double **A, int rows, int cols);
int calc_serial(double **A, int rows, int cols, int iters, double tolerance);
int calc_serial_v1(double **A, int rows, int cols, int iters, double tolerance);
int calc_gpu(double **A, int rows, int cols, int iters, double tolerance, long* time);
double verify(double **A, double **B, int rows, int cols);

__global__ void calc_gpu(int* d_arr) {
    d_arr[threadIdx.x]++;
}

int main(int argc, char * argv[])
{
    int choice = 0;
    int i;
    if (argc > 1) {
        choice = atoi(argv[1]);
    }
    int iters = 1;
    if (argc > 2) {
        iters = atoi(argv[2]);
    }
    int rows = 100, cols = 100;
    if (argc > 3) {
        rows = atoi(argv[3]);
        cols = rows;
    }

    int num_threads = 2;
    if (argc > 4) {
        num_threads = atoi(argv[4]);
    }

    double **A = (double**)malloc((rows+1)*sizeof(double*));
    double *A_mem = (double*)malloc((rows+1)*(cols+1) * sizeof(double));
    double **orig = (double**)malloc((rows+1)*sizeof(double*));
    double *orig_mem = (double*)malloc((rows+1)*(cols+1) * sizeof(double));
    for (i = 0; i < rows+1; ++i) {
        A[i] = &A_mem[i*(cols+1)];
        orig[i] = &orig_mem[i*(cols+1)];
    }

    initialize(A, rows, cols);
    initialize(orig, rows, cols);

    long startTime = 0;
    long endTime = 0;
    long diffTime = 0;

    calc_serial_v1(orig, rows, cols, iters, TOLERANCE);

    startTime = usecs();
    if (choice == 0) {
        calc_serial_v1(A, rows, cols, iters, TOLERANCE);
    	endTime = usecs();
    	diffTime = endTime-startTime;
    } else if (choice == 1) {
    } else if (choice == 2) {
	calc_gpu(A, rows, cols, iters, TOLERANCE, &diffTime);
    }

    double err = verify(orig, A, rows, cols);

    printf("Time = %ld us, with error %f", diffTime, err);

    /*dim3 grid(10);
    dim3 block(1);

    int h_arr[] = {1,2,3,4,5,6,7,8,9,10};
    int *d_arr;
    cudaMalloc(&d_arr, 10*sizeof(int));
    cudaMemcpy(d_arr, h_arr, 10*sizeof(int), cudaMemcpyHostToDevice);

    calc_gpu<<<1,10>>>(d_arr);

    cudaMemcpy(h_arr, d_arr, 10*sizeof(int), cudaMemcpyDeviceToHost);
    for (i=0; i < 10; ++i) {
        printf(" %d", h_arr[i]);
    }*/
    return 0;
}

long usecs() {
	struct timespec s = {0,0};
	clock_gettime(CLOCK_MONOTONIC, &s);
    	double time = (s.tv_sec + s.tv_nsec*(1.0e-9))*1000000;
    	return (long)(time);
}

void initialize(double **A, int rows, int cols) {
    int i,j;

    for (j = 0; j < cols+1;j++){
        A[0][j]=1.0;
    }

    for (i = 1; i < rows+1; i++){
        A[i][0]=1.0;
        for (j = 1; j < cols+1;j++) 
            A[i][j]=0.0;
    }
}

double verify(double **A, double **B, int rows, int cols) {
    double error = 0.0;

    int i,j;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            error += fabs(A[i][j] - B[i][j]);
        }
    } 
    return error;
}

int calc_serial(double **A, int rows, int cols, int iters, double tolerance) {
    int convergence = 0;
    double diff, tmp;
    int i,j;
    int for_iters;


    for (for_iters = 0; for_iters < iters; for_iters++) 
    { 
        diff = 0.0;

        for (i = 1;i < rows;i++)
        {
            for (j = 1; j < cols; j++)
            {
                tmp = A[i][j];
                A[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
                diff += fabs(A[i][j] - tmp);
            }
        }

        if (diff/((double)rows*(double)cols) < tolerance)
            convergence=1;

    }

    //printf("\nConv = %d", convergence);
    return convergence;
}

int calc_serial_v1(double **A, int rows, int cols, int iters, double tolerance) {
    int convergence = 0;
    double diff;
    int i,j;
    int for_iters;

    double** B = (double**)malloc((rows+1)*sizeof(double*));
    for (i = 0; i < rows+1; ++i) {
        B[i] = (double*)malloc((cols+1) * sizeof(double));
    }

    for (for_iters = 0; for_iters < iters; for_iters++) 
    { 
        for (i = 0; i < rows+1; ++i) {
            for (j = 0; j < cols+1; ++j) {
                B[i][j] = A[i][j];
            }
        }

        diff = 0.0;

        for (i = 1;i < rows;i++)
        {
            for (j = 1; j < cols; j++)
            {
                A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i-1][j] + B[i][j+1] + B[i+1][j]);
                diff += fabs(A[i][j] - B[i][j]);
            }
        }

        if (diff/((double)rows*(double)cols) < tolerance)
            convergence=1;

    }

    //printf("\nConv = %d", convergence);
    return convergence;
}

__global__ void calc_kernel(double* w, double* r, int rows, int cols, double tolerance) {
	int row = blockIdx.x;	
	int col = threadIdx.x;
	int idx = row*blockDim.x + col;
	if (row < rows && row > 0 && col < cols) {
		w[idx] = 0.2*(r[idx+1] + r[idx - 1] + r[(row-1)*blockDim.x + col] + r[(row+1)*blockDim.x + col]);   
	}
}

int calc_gpu(double **A, int rows, int cols, int iters, double tolerance, long *time) {
	double* d_A_curr;
	double* d_A_tmp;
	float millisec = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	cudaMalloc(&d_A_curr, (rows+1)*(cols+1)*sizeof(double));
	cudaMalloc(&d_A_tmp, (rows+1)*(cols+1)*sizeof(double));
	cudaMemcpy(d_A_curr, A[0], (rows+1)*(cols+1)*sizeof(double), cudaMemcpyHostToDevice);		
	cudaMemcpy(d_A_tmp, A[0], (rows+1)*(cols+1)*sizeof(double), cudaMemcpyHostToDevice);

	int tileSize = 100;
	int nTiles = rows / tileSize + (rows % tileSize == 0) ? 0 : 1;
	
	int for_iters;
	for (for_iters = 0; for_iters < iters; ++for_iters) {
		if (for_iters % 2 == 0) {
			calc_kernel<<<nTiles, tileSize>>>(d_A_curr, d_A_tmp, rows, cols, tolerance);
		} else {
			calc_kernel<<<nTiles, tileSize>>>(d_A_tmp, d_A_curr, rows, cols, tolerance);
		}			
	}	

	if (iters %2 == 0) {
		cudaMemcpy(A[0], d_A_curr, (rows+1)*(cols+1)*sizeof(double), cudaMemcpyDeviceToHost);		
	} else {
		cudaMemcpy(A[0], d_A_tmp, (rows+1)*(cols+1)*sizeof(double), cudaMemcpyDeviceToHost);		
	}	
	
	cudaFree(d_A_curr);		
	cudaFree(d_A_tmp);		

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisec, start, stop);
	*time = (long)(millisec*1000);
	return 0;
}
