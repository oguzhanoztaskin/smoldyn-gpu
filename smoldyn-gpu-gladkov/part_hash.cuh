/*
 * part_hash.cuh
 *
 *  Created on: Mar 15, 2011
 *      Author: denis
 */

#ifndef PART_HASH_CUH_
#define PART_HASH_CUH_

#include "thrust/device_ptr.h"
#include "thrust/scan.h"

namespace cudahelp {

/*
 * offsets - should have length of numCells + 1
 * indices - should have length of size
 */

uint* devCellOffsets = 0;

template <class T, class P>
__global__ void CalcNumberOfParticlesDev(T* mols, uint* offsets, int size,
                                         P HashFunc) {
  uint idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size) return;

  uint cell_id = HashFunc(mols[idx]);

  atomicAdd(&offsets[cell_id + 1], 1);
}

template <class T, class P>
__global__ void HashParticlesIntoCellsDev(T* mols, uint* offsets,
                                          uint* cellOffsets, uint* indices,
                                          int size, P HashFunc) {
  uint idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size) return;

  uint cell_id = HashFunc(mols[idx]);

  uint off = offsets[cell_id];

  uint off2 = atomicAdd(&cellOffsets[cell_id], 1);

  indices[off + off2] = idx;
}

__host__ void InitGrid(int numSamples, int numCells) {
  cudaMalloc(&devCellOffsets, numCells * sizeof(uint));
}

__host__ void DeinitGrid() {
  if (devCellOffsets) cudaFree(devCellOffsets);
}

template <class T, class P>
__host__ void BuildGrid(T* mols, uint* offsets, uint* indices, uint size,
                        uint numCells, P HashFunc) {
  cudaMemset(offsets, 0, (numCells + 1) * sizeof(uint));
  cudaMemset(devCellOffsets, 0, (numCells) * sizeof(uint));

  int numThreads = 256;
  int numBlocks = (size + numThreads - 1) / numThreads;

  CalcNumberOfParticlesDev<<<numBlocks, numThreads>>>(mols, offsets, size,
                                                      HashFunc);

  thrust::device_ptr<uint> dev_data(offsets);
  thrust::device_ptr<uint> dev_data_end(offsets + numCells + 1);

  thrust::inclusive_scan(dev_data, dev_data_end, dev_data);

  HashParticlesIntoCellsDev<<<numBlocks, numThreads>>>(
      mols, offsets, devCellOffsets, indices, size, HashFunc);
}

}  // namespace cudahelp

#endif /* PART_HASH_CUH_ */
