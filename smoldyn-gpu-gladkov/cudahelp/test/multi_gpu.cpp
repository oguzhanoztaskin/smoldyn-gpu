#include "multi_gpu.h"

#include "../benchmark.h"
#include "../devmanager.h"

#ifndef _MSC_VER
#include <unistd.h>
#else
#include <windows.h>
#endif

#include "kernels.h"

using namespace cudahelp;
using namespace std;

#include <string.h>

#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <iostream>

#include "utils.h"

struct locked_out {
  locked_out(std::ostream& s) : out(s) {}

  void lock() { io_mutex.lock(); }

  void unlock() { io_mutex.unlock(); }

  template <class T>
  ostream& operator<<(const T& data) {
    out << data;
    return out;
  }

 private:
  ostream& out;
  boost::mutex io_mutex;
} mt_out(cout);

struct thread_func {
  DeviceManager::DevicePtr device;

  int* data_in;
  int* data_out;

  int id;

  int size;

  bool zeroCopy;

  void operator()() {
    device->SetCurrent();
    device->EnableMapHost();

    mt_out.lock();
    mt_out << "Thread id: " << id << " using device: " << device->GetName()
           << "\n";
    mt_out.unlock();

    int* dev_ptr = 0;

    if (zeroCopy)
      cudaHostGetDevicePointer(&dev_ptr, data_out, 0);
    else
      cudaMalloc(&dev_ptr, size * sizeof(int));

    cudaMemcpy(dev_ptr, data_in, size * sizeof(int), cudaMemcpyHostToDevice);

    MultiplyInplaceGPU(dev_ptr, size);

    if (!zeroCopy) {
      cudaMemcpy(data_out, dev_ptr, size * sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(dev_ptr);
    }

    cudaThreadSynchronize();
  }
};

void do_test_multi(int* vin, int* vout, int size, CpuClockBenchmark& benchmark,
                   bool useZeroCopy) {
  thread_func tf1;
  thread_func tf2;

  tf1.device = DeviceManager::Get().GetFirstDevice();
  tf1.id = 1;

  tf1.data_in = vin;
  tf1.data_out = vout;

  tf1.size = size - (size >> 4);

  tf1.zeroCopy = useZeroCopy;

  cout << "Thread1 size is: " << tf1.size << "\n";

  tf2.device = tf1.device->GetNext();
  tf2.id = 2;
  tf2.data_in = vin + tf1.size;
  tf2.data_out = vout + tf1.size;
  tf2.size = size >> 4;

  tf2.zeroCopy = useZeroCopy;

  cout << "Thread2 size is: " << tf2.size << "\n";

  boost::thread thread1(tf1);
  boost::thread thread2(tf2);

  thread1.join();
  thread2.join();
}

void do_test_single(int* vin, int* vout, int size, CpuClockBenchmark& benchmark,
                    bool useZeroCopy) {
  DeviceManager::DevicePtr dev = DeviceManager::Get().GetMaxGFpsDevice();

  dev->SetCurrent();

  cout << "Using device: " << dev->GetName() << "\n";

  int* dev_ptr = 0;

  if (useZeroCopy) {
    dev->EnableMapHost();
    cudaHostGetDevicePointer(&dev_ptr, vout, 0);
  } else
    cudaMalloc(&dev_ptr, size * sizeof(int));

  cudaMemcpy(dev_ptr, vin, size * sizeof(int), cudaMemcpyHostToDevice);

  MultiplyInplaceGPU(dev_ptr, size);

  if (!useZeroCopy) {
    cudaMemcpy(vout, dev_ptr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);
  }

  cudaThreadSynchronize();
}

typedef void (*alloc_func)(void**, int);
typedef void (*free_func)(void*);

alloc_func AllocateData;
free_func FreeData;

void HostAlloc(void** data, int size) {
  cudaHostAlloc(data, size, cudaHostAllocDefault);
}

void HostAllocPortable(void** data, int size) {
  cudaHostAlloc(
      data, size,
      cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped);
}

void HostFree(void* data) { cudaFreeHost(data); }

void RegularAlloc(void** data, int size) { *data = malloc(size); }

void RegularFree(void* data) { free(data); }

int do_multi_gpu_test(int argc, char** argv) {
  CpuClockBenchmark benchmark;

  bool multi = true;
  bool zeroCopy = false;
  bool stdAlloc = false;

  FreeData = HostFree;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-single")) multi = false;
    if (!strcmp(argv[i], "-multi")) multi = true;
    if (!strcmp(argv[i], "-zeroCopy")) zeroCopy = true;
    if (!strcmp(argv[i], "-malloc")) stdAlloc = true;
  }

  int size = 0x10000000 >> 4;
  int *data, *output;

  cout << "Multiple devices: " << bool2str(multi)
       << " Zero copy: " << bool2str(zeroCopy)
       << " Std malloc: " << bool2str(stdAlloc) << " Size: " << size << "\n";

  if (multi && DeviceManager::Get().GetDeviceCount() < 2) {
    cout << "Less than 2 devices in the system\n";
    return 0;
  }

  AllocateData = HostAlloc;

  if (stdAlloc) AllocateData = RegularAlloc;

  if (zeroCopy) {
    DeviceManager::DevicePtr dev = DeviceManager::Get().GetMaxGFpsDevice();

    dev->SetCurrent();
    dev->EnableMapHost();
  }

  AllocateData((void**)&data, size * sizeof(int));

  if (zeroCopy)
    HostAllocPortable((void**)&output, size * sizeof(int));
  else
    AllocateData((void**)&output, size * sizeof(int));

  cout << "Generating data: " << size << "\n";

  benchmark.Start("DataGenerate");
  generate_random(data, size);
  benchmark.Stop("DataGenerate");

  cout << "Performing gpu ops\n";

  if (multi) {
    benchmark.Start("MultiGpuTest");
    do_test_multi(data, output, size, benchmark, zeroCopy);
    benchmark.Stop("MultiGpuTest");
  } else {
    benchmark.Start("SingleGpuTest");
    do_test_single(data, output, size, benchmark, zeroCopy);
    benchmark.Stop("SingleGpuTest");
  }

  cout << "Verifying data\n";

  benchmark.Start("DataVerification");

  int i = 0;
  for (; i < size; i++)
    if (output[i] / data[i] != 2) {
      cout << "TEST FAILED: " << output[i] << " " << data[i] << " " << i
           << "\n";
      break;
    }

  if (i == size) cout << "TEST PASSED\n";

  benchmark.Stop("DataVerification");

  benchmark.Print(cout);

  FreeData(data);
  FreeData(output);

  return 0;
}
