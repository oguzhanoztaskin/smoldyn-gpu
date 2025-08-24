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
 private:                       // member variables
  unsigned int *mTempKeys;      // Temporary key storage
  unsigned int *mTempValues;    // Temporary value storage
  unsigned int *mCounters;      // Histogram counters
  unsigned int *mCountersSum;   // Prefix sum of counters
  unsigned int *mBlockOffsets;  // Block offset storage
  void *mScanPlan;              // Scan plan for prefix sums
  unsigned int mMaxElements;    // Maximum number of elements
  bool mKeysOnly;               // Keys-only mode flag
  bool mInitialized;            // Initialization flag

 public:  // methods
  RadixSort(unsigned int maxElements, bool keysOnly = false);

  //------------------------------------------------------------------------
  // Destructor
  //------------------------------------------------------------------------
  ~RadixSort();

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
                 unsigned int *tempKeys, unsigned int *tempValues,
                 unsigned int *counters, unsigned int *countersSum,
                 unsigned int *blockOffsets, void *scanPlan,
                 unsigned int numElements, unsigned int keyBits, bool flipBits);

  void radixSortKeysOnly(unsigned int *keys, unsigned int *tempKeys,
                         unsigned int *counters, unsigned int *countersSum,
                         unsigned int *blockOffsets, void *scanPlan,
                         unsigned int numElements, unsigned int keyBits,
                         bool flipBits);

  void initialize();
  void cleanup();

 public:                                     // constants
  static const unsigned int CTA_SIZE = 256;  // Number of threads per block
  static const unsigned int WARP_SIZE = 32;
  static const unsigned int RADIX_BITS = 4;   // Process 4 bits per pass
  static const unsigned int RADIX_SIZE = 16;  // 2^4 = 16 bins
};

}  // namespace nvRadixSort
#endif  // __RADIXSORT_H__
