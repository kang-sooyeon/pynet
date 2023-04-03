
#ifndef GPU_DEVICE_H_
#define GPU_DEVICE_H_

#include <CL/cl.h>

namespace gpu {

class clDnnHandle_t {
 public:
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
};

class Device {
 public:
  Device();
  virtual ~Device();

  cl_int clDnnCreate(clDnnHandle_t **handle);
  cl_int clDnnDestroy(clDnnHandle_t *handle);
  clDnnHandle_t *ClDnnHandle() const;
  void GetLocalSize(size_t *localSize, int ndim);
  void Flush();

 private:
  clDnnHandle_t *clDnnHandle_;
  size_t workGroupSize[4];
};
}  // namespace gpu

#endif  // GPU_DEVICE_H_
