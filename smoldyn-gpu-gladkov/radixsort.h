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

#ifndef __RADIXSORT_H__
#define __RADIXSORT_H__

// -----------------------------------------------------------------------
// Fast CUDA Radix Sort Class
//
// The parallel radix sort algorithm implemented by this code is described
// in the following paper.
//
// Satish, N., Harris, M., and Garland, M. "Designing Efficient Sorting
// Algorithms for Manycore GPUs". In Proceedings of IEEE International
// Parallel & Distributed Processing Symposium 2009 (IPDPS 2009).
//
// -----------------------------------------------------------------------

#include <iostream>

#include "cuda_runtime_api.h"

namespace nvRadixSort {

class RadixSort {
 public:
  RadixSort(int _unused);

  //------------------------------------------------------------------------
  // Sorts input arrays of unsigned integer keys and (optional) values
  //
  // @param keys        Array of keys for data to be sorted
  // @param values      Array of values to be sorted
  // @param numElements Number of elements to be sorted.  Must be <=
  //                    maxElements passed to the constructor
  // @param keyBits     The number of bits in each key to use for ordering
  //------------------------------------------------------------------------
  void sort(unsigned int *keys, unsigned int *values, unsigned int numElements,
            unsigned int keyBits);

 private:  // helper methods
  void radixSort(unsigned int *keys, unsigned int *values,
                 unsigned int numElements);

  void radixSortKeysOnly(unsigned int *keys, unsigned int numElements);
};

}  // namespace nvRadixSort
#endif  // __RADIXSORT_H__
