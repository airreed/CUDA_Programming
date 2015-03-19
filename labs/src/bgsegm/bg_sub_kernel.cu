#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "bg_sub_kernel.h"

#define STD_T 400


  
__global__ void naiveBackgroundSubstraction(const Matrix M[], const Matrix sub,
                                             Matrix result) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;


  if(x >= sub.width || y >= sub.height) return;

  int tmp = 0;
  for(int i=0; i<5; i++) {
    tmp += M[i].elements[y * sub.width + x];
  }
  tmp = tmp / 5;

  float diff = sub.elements[y * sub.width + x] - tmp;
  
  if(powf(diff, 2) < STD_T) {
    result.elements[y * sub.width + x] = (unsigned char)255;
  }
  else {
    result.elements[y * sub.width + x] = (unsigned char)0;
  }
}

void bg_caller(const Matrix M[], const Matrix sub,  Matrix result) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid(ceil(sub.width / dimBlock.x), ceil(sub.height / dimBlock.y));
 
//  printf("dimGrid.x: %d, dimGrid.y: %d\n", dimGrid.x, dimGrid.y);
  naiveBackgroundSubstraction<<<dimGrid, dimBlock>>>(M, sub,  result);
  cudaDeviceSynchronize();
}    

Matrix AllocateDeviceMatrix(const Matrix M) {
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(unsigned char);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost) {
  Mdevice.height = Mhost.height;
  Mdevice.width = Mhost.width;
  Mdevice.pitch = Mdevice.height;
  int size = Mhost.width * Mhost.height * sizeof(unsigned char);
  cudaMemcpy(Mdevice.elements, Mhost.elements, size,
      cudaMemcpyHostToDevice);
}

void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice) {
  int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
  cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
      cudaMemcpyDeviceToHost);
}

void FreeDeviceMatrix(Matrix* M) {
  cudaFree(M->elements);
  M->elements = NULL;
}

void FreeMatrix(Matrix * M) {
  free(M->elements);
  M->elements = NULL;
}
