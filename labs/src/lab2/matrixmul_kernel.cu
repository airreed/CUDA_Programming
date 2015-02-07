/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"


////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__device__ float getElements(const Matrix A, int row, int col) {
  return A.elements[row * A.pitch + col];
}

__device__ void setElements(const Matrix A, int row, int col, float value) {
  A.elements[row * A.pitch + col] = value;
}

__device__ Matrix getSubMatrix(Matrix A, int row, int col) {
  Matrix sub;
  sub.width = BLOCK_SIZE;
  sub.height = BLOCK_SIZE;
  sub.pitch = A.pitch;
  sub.elements = &A.elements[A.pitch * BLOCK_SIZE * row
                                      + BLOCK_SIZE * col];

  return sub;
}

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(const Matrix M, const Matrix N, Matrix P)
{
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  Matrix subP = getSubMatrix(P, blockRow, blockCol);

  float valueP = 0.0f;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int i = 0;
   /**
    *  M.width / BLOCK_SIZE may be a integer, we still need to handle
    *  the rest of the computation.
    */
  for(i=0; i<(M.width+BLOCK_SIZE-1)/BLOCK_SIZE ; i++) {
    Matrix subM = getSubMatrix(M, blockRow, i);
    Matrix subN = getSubMatrix(N, i, blockCol);

    /** 
     *  shared memory to store two matrix 
     *  Add one padding column to each matrix 
     *  to avoid bank conflict.
     *  ***** there is another method to avoid
     *  ***** bank conflict that using transpose matrix
     *        to change one of the matrix to change the 
     *        reading step.
     */
    __shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float sharedN[BLOCK_SIZE+1][BLOCK_SIZE];

    /** if the position is out of the matrix, give it 0.0 */
    if(i * blockDim.x + col < M.width && blockDim.y * blockIdx.y + threadIdx.y < M.height){
      sharedM[row][col] = getElements(subM, row, col);
    }
    else {
      sharedM[row][col] = 0.0;
    }
    __syncthreads();
    if(i * blockDim.y + row < N.height && blockDim.x * blockIdx.x + threadIdx.x < N.width){
      sharedN[row][col] = getElements(subN, row, col);
    }
    else {
      sharedN[row][col] = 0.0;
    }

    __syncthreads();

    int j = 0;
    for(j = 0; j <BLOCK_SIZE; j++) {
      valueP = valueP + sharedM[row][j] * sharedN[j][col];
    }
    /** printf("counter: %d, j=%d\n", counter, j);  */

    __syncthreads();
  }
  
  /** if the position of the result matrix is in the edge. set the element */
  if(( blockDim.y * blockIdx.y + threadIdx.y < P.height) && (blockDim.x * blockIdx.x + threadIdx.x < P.width))
    setElements(subP, row, col, valueP);

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
