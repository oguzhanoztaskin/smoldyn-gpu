/*
 * devmanager.cpp
 *
 *  Created on: Aug 14, 2010
 *      Author: denis
 */

#include "devmanager.h"

#include <ostream>
#include <stdexcept>

#undef max
#undef min

namespace cudahelp {

DeviceManager* DeviceManager::instance_ = 0;

DeviceManager::Device::Device(int id, int total) : id_(id), total_(total) {
  cudaGetDeviceProperties(&deviceProp_, id_);
}

DeviceManager::Device::~Device() {}

boost::shared_ptr<DeviceManager::Device> DeviceManager::Device::GetNext()
    const {
  if (HasNext())
    return DevicePtr(new DeviceManager::Device(id_ + 1, total_));
  else
    throw std::logic_error("Invalid device requested from DeviceManager");
}

bool DeviceManager::Device::HasNext() const { return total_ - id_ > 1; }

std::string DeviceManager::Device::GetName() const { return deviceProp_.name; }

void DeviceManager::Device::SetCurrent() { cudaSetDevice(id_); }

inline std::string bool2str(bool b) { return (b) ? "yes" : "no"; }

void DeviceManager::Device::Print(std::ostream& out) const {
  out << "Name: " << deviceProp_.name << "\n"
      << "Total global memory: "
      << deviceProp_.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << " Gbytes"
      << "\n"
      << "Shared memory per block: " << deviceProp_.sharedMemPerBlock / 1024.0
      << " Kbytes" << "\n"
      << "Registers per block: " << deviceProp_.regsPerBlock << "\n"
      << "Warp size: " << deviceProp_.warpSize << "\n"
      << "Max grid dimensions: " << deviceProp_.maxGridSize[0] << "x"
      << deviceProp_.maxGridSize[1] << "x" << deviceProp_.maxGridSize[2] << "\n"
      << "Max block dimensions: " << deviceProp_.maxThreadsDim[0] << "x"
      << deviceProp_.maxThreadsDim[1] << "x" << deviceProp_.maxThreadsDim[2]
      << "\n"
      << "Max threads per block: " << deviceProp_.maxThreadsPerBlock << "\n"
      << "Total constant memory: " << deviceProp_.totalConstMem / 1024.0
      << " Kbytes" << "\n"
      << "Version: " << deviceProp_.major << "." << deviceProp_.minor << "\n"
      << "Clock rate: " << deviceProp_.clockRate / 1000.0 / 1000.0 << " GHz"
      << "\n"
      << "Multiprocessor count: " << deviceProp_.multiProcessorCount << "\n"
      << "Can map host memory: " << bool2str(deviceProp_.canMapHostMemory == 1)
      << "\n"
      << "Can overlap: " << bool2str(deviceProp_.deviceOverlap == 1) << "\n";
}

bool DeviceManager::Device::SupportOverlaps() const {
  return deviceProp_.deviceOverlap == 1;
}

bool DeviceManager::Device::SupportMapHost() const {
  return deviceProp_.canMapHostMemory == 1;
}

void DeviceManager::Device::EnableMapHost() {
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

// from cutil
int GetMaxGfpsDevid() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  int arch_cores_sm[3] = {1, 8, 32};
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount(&device_count);
  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = std::max(best_SM_arch, deviceProp.major);
    }
    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } else if (deviceProp.major <= 2) {
      sm_per_multiproc = arch_cores_sm[deviceProp.major];
    } else {
      sm_per_multiproc = arch_cores_sm[2];
    }

    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc *
                       deviceProp.clockRate;
    if (compute_perf > max_compute_perf) {
      // If we find GPU with SM major > 2, search only these
      if (best_SM_arch > 2) {
        // If our device==dest_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      } else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }
    ++current_device;
  }
  return max_perf_device;
}

DeviceManager& DeviceManager::Get() {
  if (instance_ == 0) instance_ = new DeviceManager;

  return *instance_;
}

void DeviceManager::Destroy() {
  if (instance_) delete instance_;
}

DeviceManager::DevicePtr DeviceManager::GetFirstDevice() const {
  return DeviceManager::DevicePtr(new Device(0, totalDevices_));
}

DeviceManager::DevicePtr DeviceManager::GetMaxGFpsDevice() const {
  return DeviceManager::DevicePtr(new Device(GetMaxGfpsDevid(), totalDevices_));
}

int DeviceManager::GetDeviceCount() const { return totalDevices_; }

DeviceManager::DevicePtr DeviceManager::GetCurrentDevice() const {
  int id;
  cudaGetDevice(&id);
  return DevicePtr(new Device(id, totalDevices_));
}

DeviceManager::DeviceManager() : totalDevices_(0) {
  cudaGetDeviceCount(&totalDevices_);
}

DeviceManager::~DeviceManager() {}

}  // namespace cudahelp
