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
void RadixSort::radixSortKeysOnly(unsigned int *keys,
                                  unsigned int numElements) {
  // Use Thrust for a simple implementation
  thrust::device_ptr<unsigned int> keys_ptr(keys);
  thrust::device_ptr<unsigned int> keys_end = keys_ptr + numElements;
  thrust::sort(keys_ptr, keys_end);
}

void RadixSort::radixSort(unsigned int *keys, unsigned int *values,
                          unsigned int numElements) {
  // Use Thrust for a simple implementation
  thrust::device_ptr<unsigned int> keys_ptr(keys);
  thrust::device_ptr<unsigned int> keys_end = keys_ptr + numElements;
  thrust::device_ptr<unsigned int> values_ptr(values);
  thrust::sort_by_key(keys_ptr, keys_end, values_ptr);
}

}  // namespace nvRadixSort