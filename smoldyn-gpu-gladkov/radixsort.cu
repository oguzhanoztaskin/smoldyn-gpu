/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "radixsort.h"

namespace nvRadixSort {

// Simple implementation using Thrust for the member functions
void RadixSort::radixSortKeysOnly(unsigned int *keys, unsigned int *tempKeys,
                                  unsigned int *counters,
                                  unsigned int *countersSum,
                                  unsigned int *blockOffsets, void *scanPlan,
                                  unsigned int numElements,
                                  unsigned int keyBits, bool flipBits) {
  // Use Thrust for a simple implementation
  thrust::device_ptr<unsigned int> keys_ptr(keys);
  thrust::device_ptr<unsigned int> keys_end = keys_ptr + numElements;
  thrust::sort(keys_ptr, keys_end);
}

void RadixSort::radixSort(unsigned int *keys, unsigned int *values,
                          unsigned int *tempKeys, unsigned int *tempValues,
                          unsigned int *counters, unsigned int *countersSum,
                          unsigned int *blockOffsets, void *scanPlan,
                          unsigned int numElements, unsigned int keyBits,
                          bool flipBits) {
  // Use Thrust for a simple implementation
  thrust::device_ptr<unsigned int> keys_ptr(keys);
  thrust::device_ptr<unsigned int> keys_end = keys_ptr + numElements;
  thrust::device_ptr<unsigned int> values_ptr(values);
  thrust::sort_by_key(keys_ptr, keys_end, values_ptr);
}

// Wrapper functions callable from C++
void radixSortKeysOnlyImpl(unsigned int *keys, unsigned int *tempKeys,
                           unsigned int *counters, unsigned int *countersSum,
                           unsigned int *blockOffsets, void *scanPlan,
                           unsigned int numElements, unsigned int keyBits,
                           bool flipBits) {
  // Use Thrust for a simple implementation
  thrust::device_ptr<unsigned int> keys_ptr(keys);
  thrust::device_ptr<unsigned int> keys_end = keys_ptr + numElements;
  thrust::sort(keys_ptr, keys_end);
}

void radixSortImpl(unsigned int *keys, unsigned int *values,
                   unsigned int *tempKeys, unsigned int *tempValues,
                   unsigned int *counters, unsigned int *countersSum,
                   unsigned int *blockOffsets, void *scanPlan,
                   unsigned int numElements, unsigned int keyBits,
                   bool flipBits) {
  // Use Thrust for a simple implementation
  thrust::device_ptr<unsigned int> keys_ptr(keys);
  thrust::device_ptr<unsigned int> keys_end = keys_ptr + numElements;
  thrust::device_ptr<unsigned int> values_ptr(values);
  thrust::sort_by_key(keys_ptr, keys_end, values_ptr);
}

// CUDA kernel for radix sort pass (keys only)
__global__ void radixSortKeysKernel(unsigned int *keysIn, unsigned int *keysOut,
                                    unsigned int *histogram,
                                    unsigned int *blockOffsets,
                                    unsigned int numElements,
                                    unsigned int startBit) {
  extern __shared__ unsigned int sdata[];

  unsigned int *sRadix = sdata;
  unsigned int *sKeys = (unsigned int *)&sdata[RadixSort::RADIX_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int blockId = blockIdx.x;
  unsigned int globalId = blockId * blockDim.x + tid;

  // Initialize shared memory counters
  if (tid < RadixSort::RADIX_SIZE) {
    sRadix[tid] = 0;
  }
  __syncthreads();

  // Load keys and count radix values
  unsigned int key = 0;
  unsigned int radixVal = 0;
  if (globalId < numElements) {
    key = keysIn[globalId];
    sKeys[tid] = key;
    radixVal = (key >> startBit) & (RadixSort::RADIX_SIZE - 1);
    atomicAdd(&sRadix[radixVal], 1);
  }
  __syncthreads();

  // Write histogram to global memory
  if (tid < RadixSort::RADIX_SIZE) {
    histogram[blockId * RadixSort::RADIX_SIZE + tid] = sRadix[tid];
  }
  __syncthreads();

  // Compute block-local scan
  unsigned int localScan[RadixSort::RADIX_SIZE];
  if (tid == 0) {
    localScan[0] = 0;
    for (int i = 1; i < RadixSort::RADIX_SIZE; i++) {
      localScan[i] = localScan[i - 1] + sRadix[i - 1];
    }
    for (int i = 0; i < RadixSort::RADIX_SIZE; i++) {
      sRadix[i] = localScan[i];
    }
  }
  __syncthreads();

  // Scatter keys to output
  if (globalId < numElements) {
    unsigned int globalOffset =
        blockOffsets[blockId * RadixSort::RADIX_SIZE + radixVal];
    unsigned int localOffset = atomicAdd(&sRadix[radixVal], 1);
    keysOut[globalOffset + localOffset] = key;
  }
}

// CUDA kernel for radix sort pass (keys and values)
__global__ void radixSortPairsKernel(
    unsigned int *keysIn, unsigned int *keysOut, unsigned int *valuesIn,
    unsigned int *valuesOut, unsigned int *histogram,
    unsigned int *blockOffsets, unsigned int numElements,
    unsigned int startBit) {
  extern __shared__ unsigned int sdata[];

  unsigned int *sRadix = sdata;
  unsigned int *sKeys = (unsigned int *)&sdata[RadixSort::RADIX_SIZE];
  unsigned int *sValues = (unsigned int *)&sKeys[blockDim.x];

  unsigned int tid = threadIdx.x;
  unsigned int blockId = blockIdx.x;
  unsigned int globalId = blockId * blockDim.x + tid;

  // Initialize shared memory counters
  if (tid < RadixSort::RADIX_SIZE) {
    sRadix[tid] = 0;
  }
  __syncthreads();

  // Load keys/values and count radix values
  unsigned int key = 0;
  unsigned int value = 0;
  unsigned int radixVal = 0;
  if (globalId < numElements) {
    key = keysIn[globalId];
    value = valuesIn[globalId];
    sKeys[tid] = key;
    sValues[tid] = value;
    radixVal = (key >> startBit) & (RadixSort::RADIX_SIZE - 1);
    atomicAdd(&sRadix[radixVal], 1);
  }
  __syncthreads();

  // Write histogram to global memory
  if (tid < RadixSort::RADIX_SIZE) {
    histogram[blockId * RadixSort::RADIX_SIZE + tid] = sRadix[tid];
  }
  __syncthreads();

  // Compute block-local scan
  unsigned int localScan[RadixSort::RADIX_SIZE];
  if (tid == 0) {
    localScan[0] = 0;
    for (int i = 1; i < RadixSort::RADIX_SIZE; i++) {
      localScan[i] = localScan[i - 1] + sRadix[i - 1];
    }
    for (int i = 0; i < RadixSort::RADIX_SIZE; i++) {
      sRadix[i] = localScan[i];
    }
  }
  __syncthreads();

  // Scatter keys and values to output
  if (globalId < numElements) {
    unsigned int globalOffset =
        blockOffsets[blockId * RadixSort::RADIX_SIZE + radixVal];
    unsigned int localOffset = atomicAdd(&sRadix[radixVal], 1);
    keysOut[globalOffset + localOffset] = key;
    valuesOut[globalOffset + localOffset] = value;
  }
}

}  // namespace nvRadixSort