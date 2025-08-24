#include "../common.h"
#include "../random/rndfast.cuh"
#include "kernels.h"

__global__ void TestFastRngsGPU(int* data, int count) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;

  uint4 state = cudahelp::rand::GetFastRngState(tid);

  for (int idx = tid; idx < count; idx += numThreads)
    data[idx] = cudahelp::rand::RngXor128(state);

  cudahelp::rand::SaveFastRngState(tid, state);
}

__global__ void TestFastRngsClassGPU(int* data, int count) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;

  cudahelp::rand::Xor128Generator gen(idx);

  for (; idx < count; idx += numThreads) data[idx] = gen.GetInt();
}

void TestFastRngs(int* data, int count, int rngs) {
  int numThreads = 256;
  int numBlocks = cudahelp::GetNumberOfBlocks(numThreads, rngs);

  TestFastRngsGPU<<<numBlocks, numThreads>>>(data, count);

  cudaThreadSynchronize();

  cudahelp::CheckCUDAError("TestFastRngs");
}

void TestFastRngsClass(int* data, int count, int rngs) {
  int numThreads = 256;
  int numBlocks = cudahelp::GetNumberOfBlocks(numThreads, rngs);

  TestFastRngsClassGPU<<<numBlocks, numThreads>>>(data, count);

  cudaThreadSynchronize();

  cudahelp::CheckCUDAError("TestFastRngsClass");
}
