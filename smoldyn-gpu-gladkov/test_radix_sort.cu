#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "radixsort.h"

int main() {
  const int N = 1000;
  std::cout << "Testing RadixSort implementation with " << N << " elements..."
            << std::endl;

  // Generate random test data
  std::vector<unsigned int> keys(N);
  std::vector<unsigned int> values(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dis(0, 0xFFFFFFFF);

  for (int i = 0; i < N; i++) {
    keys[i] = dis(gen);
    values[i] = i;  // Values are just indices
  }

  // Copy reference data for verification
  std::vector<unsigned int> ref_keys = keys;
  std::vector<unsigned int> ref_values = values;

  // Sort reference data with standard sort
  std::vector<std::pair<unsigned int, unsigned int>> ref_pairs;
  for (int i = 0; i < N; i++) {
    ref_pairs.push_back({ref_keys[i], ref_values[i]});
  }
  std::sort(ref_pairs.begin(), ref_pairs.end());

  // Allocate GPU memory
  unsigned int *d_keys, *d_values;
  cudaMalloc(&d_keys, N * sizeof(unsigned int));
  cudaMalloc(&d_values, N * sizeof(unsigned int));

  // Copy data to GPU
  cudaMemcpy(d_keys, keys.data(), N * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values.data(), N * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  // Create RadixSort instance and sort
  nvRadixSort::RadixSort sorter(N, false);
  sorter.sort(d_keys, d_values, N, 32);

  // Copy results back
  std::vector<unsigned int> result_keys(N);
  std::vector<unsigned int> result_values(N);
  cudaMemcpy(result_keys.data(), d_keys, N * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(result_values.data(), d_values, N * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  // Verify results
  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (result_keys[i] != ref_pairs[i].first ||
        result_values[i] != ref_pairs[i].second) {
      correct = false;
      std::cout << "Mismatch at index " << i << ": got (" << result_keys[i]
                << ", " << result_values[i] << "), expected ("
                << ref_pairs[i].first << ", " << ref_pairs[i].second << ")"
                << std::endl;
      break;
    }
  }

  if (correct) {
    std::cout << "✓ RadixSort implementation is working correctly!"
              << std::endl;
  } else {
    std::cout << "✗ RadixSort implementation has errors!" << std::endl;
  }

  // Test keys-only sorting
  cudaMemcpy(d_keys, keys.data(), N * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  sorter.sort(d_keys, nullptr, N, 32);

  cudaMemcpy(result_keys.data(), d_keys, N * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  std::sort(ref_keys.begin(), ref_keys.end());

  bool keys_correct = true;
  for (int i = 0; i < N; i++) {
    if (result_keys[i] != ref_keys[i]) {
      keys_correct = false;
      std::cout << "Keys-only mismatch at index " << i << ": got "
                << result_keys[i] << ", expected " << ref_keys[i] << std::endl;
      break;
    }
  }

  if (keys_correct) {
    std::cout << "✓ Keys-only sorting is working correctly!" << std::endl;
  } else {
    std::cout << "✗ Keys-only sorting has errors!" << std::endl;
  }

  // Cleanup
  cudaFree(d_keys);
  cudaFree(d_values);

  return 0;
}
