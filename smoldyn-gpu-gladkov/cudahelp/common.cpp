#include "common.h"

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace cudahelp {

bool CheckCUDAError(const char* msg) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    std::cerr << "Cuda error in " << msg << " " << cudaGetErrorString(err)
              << "\n";
    return false;
  }

  return true;
}

int GetNumberOfBlocks(int numThreads, int numSamples) {
  int numBlocks = numSamples / numThreads;

  if (numSamples % numThreads) numBlocks++;

  return numBlocks;
}

void CheckCUDAErrorAndThrow(const char* msg) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    std::ostringstream out;
    out << "Cuda error in " << msg << " " << cudaGetErrorString(err);

    throw std::runtime_error(out.str());
  }
}

}  // namespace cudahelp
