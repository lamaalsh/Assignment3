#include <stdio.h>
#include <cuda_runtime.h>
_global_ void MatrixMulKernel(float* M, float* N, float* P, int Mrow,int Mcol,int Ncol) {

  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < Mrow) && (Col < Ncol)) {
    float Pvalue = 0;

    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < Mcol; ++k) {
      Pvalue += M[Row * Mrow + k] * N[k * Mcol + Col];
    }

    P[Row * Ncol + Col] = Pvalue;
  }
}

int main() {
  int rowM = 1000; //change to a very high number (like >700)
  int colM = 1100; //change to a very high number (like >700)
  int colN = 1200; //change to a very high number (like >700)

  // Allocate memory for matrices M, N, and P on the host
  float *M, *N, *P;
  N = (float*)malloc(colM*colN*sizeof(float));
  M = (float*)malloc(rowM*colM*sizeof(float));
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
  dim3 dimBlock(16,16); //for scalability you need to try with (16,16) , (8,16) and (16,8)
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
