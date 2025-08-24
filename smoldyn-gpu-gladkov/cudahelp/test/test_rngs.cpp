#include "test_rngs.h"

#include <iostream>

#include "../benchmark.h"
#include "../devmanager.h"
#include "../random/mtwister.h"
#include "../random/rndfast.h"
#include "kernels.h"

using namespace std;

using namespace cudahelp;

void test_rngs_gpu() {
  std::cout << "Testing Mersenne Twister RNG\n";

  DeviceManager::DevicePtr dev = DeviceManager::Get().GetMaxGFpsDevice();
  dev->SetCurrent();

  const int rngsCount = 65536;
  const int numSamples = 0x10000000 >> 3;

  int* data = 0;

  if (!rand::InitGPUTwisters("data/MersenneTwister.dat", rngsCount, 999)) {
    std::cout << "Can't init twisters\n";
    return;
  }

  cudaMalloc(&data, numSamples * sizeof(int));

  std::cout << "Testing	" << numSamples << " samples\n";

  CudaBenchmark bench;

  bench.Start("RandGen");

  TestMersenneTwisterClass(data, numSamples, rngsCount);

  bench.Stop("RandGen");

  std::cout << numSamples << " generated in " << bench.GetValue("RandGen")
            << " ms\n";

  cudaFree(data);

  rand::DeinitGPUTwisters();

  cudaThreadExit();
}

void test_fast_rngs_gpu() {
  std::cout << "Testing Fast RNGs\n";

  DeviceManager::DevicePtr dev = DeviceManager::Get().GetMaxGFpsDevice();
  dev->SetCurrent();

  const int rngsCount = 65536;
  const int numSamples = 0x10000000 >> 3;

  int* data = 0;

  if (!rand::InitFastRngs(rngsCount)) {
    std::cout << "Can't init RNGs\n";
    return;
  }

  cudaMalloc(&data, numSamples * sizeof(int));

  std::cout << "Testing	" << numSamples << " samples\n";

  CudaBenchmark bench;

  bench.Start("FastRandGen");

  TestFastRngs(data, numSamples, rngsCount);

  bench.Stop("FastRandGen");

  std::cout << numSamples << " generated in " << bench.GetValue("FastRandGen")
            << " ms\n";

  cudaFree(data);

  rand::DeinitFastRngs();

  cudaThreadExit();
}
