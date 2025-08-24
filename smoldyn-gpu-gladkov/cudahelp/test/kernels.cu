#include "kernels.h"

__global__ void MultiplyInplaceDev(int* data, int size) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) data[tid] *= 2;
}

__global__ void MultiplyDev(int* in_data, int* out_data, int size) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size) {
    int data = in_data[tid];
    out_data[tid] = data * 2;
  }
}

void MultiplyInplaceGPU(int* data, int size) {
  const int numThreads = 512;
  const int numBlocks = (size + numThreads - 1) / numThreads;

  MultiplyInplaceDev<<<numBlocks, numThreads>>>(data, size);
}

void MultiplyGPU(int* in_data, int* out_data, int size) {
  const int numThreads = 512;
  const int numBlocks = (size + numThreads - 1) / numThreads;

  MultiplyDev<<<numBlocks, numThreads>>>(in_data, out_data, size);
}

void MultiplyGPUStreams(int* in_data, int* out_data, int size, cudaStream_t s) {
  const int numThreads = 512;
  const int numBlocks = (size + numThreads - 1) / numThreads;

  MultiplyDev<<<numBlocks, numThreads, 0, s>>>(in_data, out_data, size);
}
