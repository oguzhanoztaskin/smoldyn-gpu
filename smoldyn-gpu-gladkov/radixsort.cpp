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

#include "radixsort.h"

#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>

namespace nvRadixSort {

// RadixSort class implementation
RadixSort::RadixSort(int _unused) {}

void RadixSort::sort(unsigned int *keys, unsigned int *values,
                     unsigned int numElements, unsigned int keyBits) {
  if (values == nullptr) {
    radixSortKeysOnly(keys, numElements);
  } else {
    radixSort(keys, values, numElements);
  }
}

// Note: Implementation is in radixsort.cu due to CUDA/Thrust requirements

}  // namespace nvRadixSort