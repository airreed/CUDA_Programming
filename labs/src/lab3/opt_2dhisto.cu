#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void naiveHistKernel(uint32_t *input , unsigned int *pHist, 
                                size_t height, size_t width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < HISTO_HEIGHT * HISTO_WIDTH) {
    pHist[i] = 0;
  }
  __syncthreads();
  int stride = blockDim.x * gridDim.x;
  while(i < height * 4096) {
    if(i % ((INPUT_WIDTH + 128) & 0xFFFFFF80) < width)
      atomicAdd(&(pHist[input[i]]),1);
    i += stride;
  }
  __syncthreads();
}

void opt_2dhisto(uint32_t *d_input, unsigned int *d_pHist)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

  naiveHistKernel<<<INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80) / 1024 , 1024>>>(d_input, d_pHist, INPUT_HEIGHT, INPUT_WIDTH);
  /***
    * GPU kernel lauch is *asynchronous*, 
    * it will immediately return to CPU thread 
    * when CPU called GPU kernel, which means
    * it will get back to CPU before GPU finish works.
    * So we need `cudaDeviceSynchronize()` to return the 
    * actual time of GPU running.
    */
  cudaDeviceSynchronize();

}

/* Include below the implementation of any other functions you need */
void* AllocateDeviceMemory(size_t size) {
  void * addr;
  cudaMalloc(&addr, size);
  return addr;
}

void CopyToDeviceMemory(void *d_input, void *input, size_t size) {
  cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
}

void CopyFromDeviceMemory(void * output,  void * d_output,
                          size_t size) {
  cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
}
  
void FreeDeviceMemory(void *d) {
  cudaFree(d);
}
