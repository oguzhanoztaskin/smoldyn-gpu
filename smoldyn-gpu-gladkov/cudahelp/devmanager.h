/*
 * devmanager.h
 *
 *  Created on: Aug 14, 2010
 *      Author: denis
 */

#ifndef DEVMANAGER_H_
#define DEVMANAGER_H_

#ifdef _MSC_VER
#define BOOST_USE_WINDOWS_H
#endif

#include <cuda_runtime.h>

#include <boost/shared_ptr.hpp>
#include <string>

namespace cudahelp {
class DeviceManager {
 public:
  class Device {
   public:
    Device(int id, int total);
    ~Device();

    boost::shared_ptr<Device> GetNext() const;
    bool HasNext() const;

    std::string GetName() const;

    bool SupportMapHost() const;
    bool SupportOverlaps() const;

    void EnableMapHost();

    void SetCurrent();

    void Print(std::ostream& out) const;

    const cudaDeviceProp& GetProps() const { return deviceProp_; }

   private:
    int id_;
    int total_;
    cudaDeviceProp deviceProp_;
  };

  typedef boost::shared_ptr<Device> DevicePtr;

  static DeviceManager& Get();
  static void Destroy();

  DevicePtr GetFirstDevice() const;
  DevicePtr GetMaxGFpsDevice() const;
  DevicePtr GetCurrentDevice() const;

  int GetDeviceCount() const;

 private:
  static DeviceManager* instance_;

  int totalDevices_;

  DeviceManager();
  ~DeviceManager();
};
}  // namespace cudahelp

#endif /* DEVMANAGER_H_ */
