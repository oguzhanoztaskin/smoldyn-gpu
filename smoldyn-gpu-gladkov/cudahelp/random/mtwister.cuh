#ifndef MTWISTER_CUH
#define MTWISTER_CUH

#include <stdio.h>

#include "../common.h"

#define DCMT_SEED 4172
#define MT_RNG_PERIOD 607

#define MT_MM 9
#define MT_NN 19
#define MT_WMASK 0xFFFFFFFFU
#define MT_UMASK 0xFFFFFFFEU
#define MT_LMASK 0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18

namespace cudahelp {
namespace rand {

typedef struct {
  unsigned int matrix_a;
  unsigned int mask_b;
  unsigned int mask_c;
  unsigned int seed;
} mt_struct_stripped_t;

struct mt_state_t {
  int iState;
  unsigned int mti1;
  unsigned int mt[MT_NN];
};

__device__ mt_struct_stripped_t* devMT = 0;
__device__ mt_state_t* devMTState = 0;

mt_struct_stripped_t* h_devMT = 0;
mt_state_t* h_devMTState = 0;

unsigned long lcg_rand_cpu(unsigned long a) {
  return (a * 279470273UL) % 4294967291UL;
}

__device__ void InitTwisterState(mt_state_t& mts,
                                 mt_struct_stripped_t& config) {
  mts.mt[0] = config.seed;

  for (int i = 1; i < MT_NN; i++)
    mts.mt[i] =
        (1812433253U * (mts.mt[i - 1] ^ (mts.mt[i - 1] >> 30)) + i) & MT_WMASK;

  mts.iState = 0;
  mts.mti1 = mts.mt[0];
}

__global__ void InitMersenneTwistersDev(int count) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= count) return;

  mt_state_t mts = devMTState[idx];

  mt_struct_stripped_t config = devMT[idx];

  InitTwisterState(mts, config);

  devMTState[idx] = mts;
}

__global__ void InitGlobalArray(mt_struct_stripped_t* local_devMT,
                                mt_state_t* local_devMTState) {
  devMT = local_devMT;
  devMTState = local_devMTState;
}

__host__ bool InitGPUTwisters(const char* fname, int rngsCount,
                              unsigned int seed) {
  FILE* f = fopen(fname, "rb");
  if (!f) {
    printf("initMTGPU(): failed to open %s\n", fname);
    return false;
  }

  mt_struct_stripped_t* h_MT = new mt_struct_stripped_t[rngsCount];

  if (!fread(h_MT, sizeof(mt_struct_stripped_t) * rngsCount, 1, f)) {
    printf("initMTGPU(): failed to load %s\n", fname);

    fclose(f);

    return false;
  }

  fclose(f);

  for (int i = 0; i < rngsCount; i++) {
    seed = lcg_rand_cpu(seed);
    h_MT[i].seed = seed;
  }

  cudaMalloc(&h_devMT, sizeof(mt_struct_stripped_t) * rngsCount);
  cudaMalloc(&h_devMTState, sizeof(mt_state_t) * rngsCount);

  cudaMemcpy(h_devMT, h_MT, sizeof(mt_struct_stripped_t) * rngsCount,
             cudaMemcpyHostToDevice);

  delete[] h_MT;

  InitGlobalArray<<<1, 1>>>(h_devMT, h_devMTState);

  cudaThreadSynchronize();

  int numThreads = 256;
  int numBlocks = GetNumberOfBlocks(numThreads, rngsCount);

  InitMersenneTwistersDev<<<numBlocks, numThreads>>>(rngsCount);

  cudaThreadSynchronize();

  return CheckCUDAError("InitGPUTwisters");
}

__host__ void DeinitGPUTwisters() {
  cudaFree(h_devMT);
  cudaFree(h_devMTState);
}

__device__ inline mt_state_t& GetMTState(int i) { return devMTState[i]; }

__device__ inline void SaveMTState(int idx, mt_state_t& state) {
  devMTState[idx] = state;
}

__device__ inline mt_struct_stripped_t& GetMTConfig(int i) { return devMT[i]; }

__device__ unsigned int MTwisterRndInt(mt_state_t& mts,
                                       mt_struct_stripped_t& config) {
  int iState1, iStateM;
  unsigned int mti, mtiM, x;

  iState1 = mts.iState + 1;
  iStateM = mts.iState + MT_MM;

  if (iState1 >= MT_NN) iState1 -= MT_NN;

  if (iStateM >= MT_NN) iStateM -= MT_NN;

  mti = mts.mti1;
  mts.mti1 = mts.mt[iState1];
  mtiM = mts.mt[iStateM];

  x = (mti & MT_UMASK) | (mts.mti1 & MT_LMASK);
  x = mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
  mts.mt[mts.iState] = x;
  mts.iState = iState1;

  // Tempering transformation
  x ^= (x >> MT_SHIFT0);
  x ^= (x << MT_SHIFTB) & config.mask_b;
  x ^= (x << MT_SHIFTC) & config.mask_c;
  x ^= (x >> MT_SHIFT1);

  return x;
}

__device__ inline float MTwisterRndFloat(mt_state_t& mts,
                                         mt_struct_stripped_t& config) {
  return ((float)MTwisterRndInt(mts, config) + 1.0f) / 4294967296.0f;
}

class MTGenerator {
 public:
  __device__ MTGenerator(int i)
      : state(devMTState[i]), config(devMT[i]), idx(i) {}
  __device__ ~MTGenerator() { devMTState[idx] = state; }

  __device__ inline unsigned int GetInt() {
    return MTwisterRndInt(state, config);
  }

  __device__ inline float GetFloat() { return MTwisterRndFloat(state, config); }

  __device__ inline unsigned int GetInt(unsigned int a, unsigned int b) {
    return a + MTwisterRndInt(state, config) % (b - a);
  }

  __device__ inline unsigned int GetInt(unsigned int m) {
    return MTwisterRndInt(state, config) % m;
  }

 private:
  mt_state_t state;
  mt_struct_stripped_t config;
  int idx;
};
}  // namespace rand
}  // namespace cudahelp
#endif
