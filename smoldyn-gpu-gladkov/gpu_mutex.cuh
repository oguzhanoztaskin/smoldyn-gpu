/*
 * gpu_mutex.cuh
 *
 *  Created on: Mar 8, 2011
 *      Author: denis
 */

#ifndef GPU_MUTEX_CUH_
#define GPU_MUTEX_CUH_

namespace cudahelp {

__device__ uint* devMutex = 0;
__device__ uint* devMutexThreads = 0;

uint* devHostMutex = 0;
uint* devHostMutexThreads = 0;

__global__ void InitMutexPointers(uint* m, uint* t) {
  devMutex = m;
  devMutexThreads = t;
}

__host__ void InitMutexes(int size) {
  cudaMalloc(&devHostMutex, sizeof(uint) * size);
  cudaMalloc(&devHostMutexThreads, sizeof(uint) * size);

  cudaMemset(devHostMutex, 0, sizeof(uint) * size);
  cudaMemset(devHostMutexThreads, 0, sizeof(uint) * size);

  InitMutexPointers<<<1, 1>>>(devHostMutex, devHostMutexThreads);
}

__host__ void DeinitMutexes() {
  if (devHostMutex) cudaFree(devHostMutex);

  if (devHostMutexThreads) cudaFree(devHostMutexThreads);
}

class GPUMutex {
 public:
  __device__ GPUMutex(int mid, int tid) : mutexid(mid), threadid(tid) {
    locked = Lock();
  }

  __device__ ~GPUMutex() { Unlock(); }

  __device__ bool Locked() const { return locked; }

 private:
  GPUMutex(const GPUMutex& m) {}
  GPUMutex operator=(const GPUMutex& m) { return *this; }

  __device__ bool Lock() {
    if (atomicCAS(&devMutex[mutexid], 0, 1) == 0) {
      atomicExch(&devMutexThreads[mutexid], threadid);

      return true;
    }

    return false;
  }

  __device__ void Unlock() {
    if (devMutexThreads[mutexid] == threadid) atomicExch(&devMutex[mutexid], 0);
  }

  int mutexid;
  int threadid;
  bool locked;
};

}  // namespace cudahelp

#endif /* GPU_MUTEX_CUH_ */
