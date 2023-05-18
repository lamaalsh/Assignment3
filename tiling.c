#include <stdio.h> 
#define TILE_SIZE 16
#include <cuda_runtime.h>
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Mrow, int Mcol, int Ncol) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float M_shared[TILE_SIZE][TILE_SIZE];
  __shared__ float N_shared[TILE_SIZE][TILE_SIZE];

  float Pvalue = 0;

  // Loop over tiles of input matrices
  for (int t = 0; t < (Mcol - 1) / TILE_SIZE + 1; ++t) {
    // Load tile of matrix M into shared memory
    if (Row < Mrow && t * TILE_SIZE + threadIdx.x < Mcol) {
      M_shared[threadIdx.y][threadIdx.x] = M[Row * Mcol + t * TILE_SIZE + threadIdx.x];
    } else {
      M_shared[threadIdx.y][threadIdx.x] = 0;
    }

    // Load tile of matrix N into shared memory
    if (Col < Ncol && t * TILE_SIZE + threadIdx.y < Mcol) {
      N_shared[threadIdx.y][threadIdx.x] = N[(t * TILE_SIZE + threadIdx.y) * Ncol + Col];
    } else {
      N_shared[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    // Multiply the tiles in shared memory
    for (int k = 0; k < TILE_SIZE; ++k) {
      Pvalue += M_shared[threadIdx.y][k] * N_shared[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Store the result in matrix P
  if (Row < Mrow && Col < Ncol) {
    P[Row * Ncol + Col] = Pvalue;
  }
}

int main() {
  int rowM = 1000; //change to a very high number (like >700)
  int colM = 1100; //change to a very high number (like >700)
  int colN = 1200; //change to a very high number (like >700)

  // Allocate memory for matrices M, N, and P on the host
  float *M, *N, *P;
  M = (float*)malloc(rowM*colM*sizeof(float));
  N = (float*)malloc(colM*colN*sizeof(float));
  P = (float*)malloc(rowM*colN*sizeof(float));

  // Initialize matrices M and N with random values
  for (int i = 0; i < rowM; ++i) {
    for (int j = 0; j < colM; ++j) {
      M[i * rowM + j] = rand() % 10;
    }
  }

  for(int i=0; i<colM;i++)
  {
    for(int j=0; j<colN; j++)
    {
      N[i * colM + j] = rand() % 10;
    }
  }
  

  // Allocate memory for matrices M, N, and P on the device
  float *d_M, *d_N, *d_P;
  cudaMalloc(&d_M, (rowM*colM)*sizeof(float));
  cudaMalloc(&d_N, (colM*colN)*sizeof(float));
  cudaMalloc(&d_P, (rowM*colN)*sizeof(float));

  // Copy matrices M and N from host to device
  cudaMemcpy(d_M, M, sizeof(float)*(rowM*colM), cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, sizeof(float)*(colM*colN), cudaMemcpyHostToDevice);

  // Define block and grid sizes
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  dim3 dimGrid((colN + dimBlock.x - 1) / dimBlock.x, (rowM + dimBlock.y - 1) / dimBlock.y);


  //time calculation
  cudaEvent_t start, end;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start,0 );

  // Call the kernel function
  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rowM, colM, colN );


  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);

  cudaEventDestroy(start);   
  cudaEventDestroy(end);
  // Copy matrix P from device to host
  cudaMemcpy(P, d_P, (rowM*colN)*sizeof(float), cudaMemcpyDeviceToHost);

  printf("time elapsed: %.8lf milliseconds\n", time);
  // Free memory on host and device
  free(M);
  free(N);
  free(P);
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
 
    return 0;
}
