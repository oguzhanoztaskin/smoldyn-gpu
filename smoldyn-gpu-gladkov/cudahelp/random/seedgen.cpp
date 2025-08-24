#include "seedgen.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <vector>

#ifndef _MSC_VER

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

uint4* GenerateRandomSeedsGPUUint4(int size) {
  int ff = open("/dev/urandom", O_RDONLY);

  if (!ff) {
    printf("Can't open /dev/urandom\n");
    return 0;
  }

  std::vector<uint4> cpuSeeds(size);

  if (size * 16 != read(ff, &cpuSeeds[0], size * 16)) {
    printf("Can't read random seeds\n");
    return 0;
  }

  close(ff);

  for (uint i = 0; i < size; i++) {
    cpuSeeds[i].x %= (4294967296 - 129);
    cpuSeeds[i].x += 128;

    cpuSeeds[i].y %= (4294967296 - 129);
    cpuSeeds[i].y += 128;

    cpuSeeds[i].z %= (4294967296 - 129);
    cpuSeeds[i].z += 128;

    cpuSeeds[i].w %= (4294967296 - 129);
    cpuSeeds[i].w += 128;
  }

  uint4* seeds = 0;

  cudaMalloc(&seeds, size * sizeof(uint4));

  cudaMemcpy(seeds, &cpuSeeds[0], size * sizeof(uint4), cudaMemcpyHostToDevice);

  return seeds;
}

#else
#include <tchar.h>
#include <windows.h>

uint4* GenerateRandomSeedsGPUUint4(int size) {
  std::vector<uint4> cpuSeeds(size);

  // Use RtlGenRandom function (SystemFunction036)
  HMODULE dll = LoadLibrary(_T("Advapi32.dll"));
  if (dll == NULL) {
    printf("Can't read random seeds: can't load Advapi32.dll\n");
    return 0;
  }

  typedef BOOLEAN(WINAPI * RtlFunc_ptr)(void*, ULONG);

  RtlFunc_ptr RtlFunc = (RtlFunc_ptr)GetProcAddress(dll, "SystemFunction036");

  if (RtlFunc == NULL) {
    printf("Can't read random seeds: can't find RtlGenRandom address\n");
    return 0;
  }

  if (!(RtlFunc)(&cpuSeeds[0], size * sizeof(uint4))) {
    FreeLibrary(dll);
    printf("Can't read random seeds: can't invoke RtlGenRandom function\n");
    return 0;
  }

  FreeLibrary(dll);

  uint4* seeds = 0;

  cudaMalloc(&seeds, size * sizeof(uint4));

  cudaMemcpy(seeds, &cpuSeeds[0], size * sizeof(uint4), cudaMemcpyHostToDevice);

  return seeds;
}

#endif
