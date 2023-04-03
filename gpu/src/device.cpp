#include "device.h"

#include <util.h>

#include <fstream>
#include <iostream>

namespace gpu {
static int convertToString(const char *filename, std::string &s) {
  size_t size;
  char *str;
  std::fstream f(filename, (std::fstream::in | std::fstream::binary));

  if (f.is_open()) {
    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);
    str = new char[size + 1];
    if (!str) {
      f.close();
      return 0;
    }
    f.read(str, fileSize);
    f.close();
    str[size] = '\0';
    s = str;
    delete[] str;
    return 0;
  } else {
    std::cout << "convertToString error" << std::endl;
    exit(-1);
  }
}

Device::Device() {
  clDnnHandle_ = nullptr;
  CheckCl(clDnnCreate(&clDnnHandle_));

  size_t nameSize = 0;
  clGetDeviceInfo(clDnnHandle_->device, CL_DEVICE_NAME, 0, NULL, &nameSize);

  char *name = (char *)malloc(nameSize);
  clGetDeviceInfo(clDnnHandle_->device, CL_DEVICE_NAME, nameSize, name, NULL);
  std::cout << "device name : " << name << std::endl;
  free(name);

  cl_uint maxComputeUnits = 0;
  clGetDeviceInfo((clDnnHandle_)->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                  sizeof(cl_uint), &maxComputeUnits, NULL);
  std::cout << "Max compute units : " << maxComputeUnits << std::endl;

  size_t maxWorkGroupSize = 0;
  clGetDeviceInfo((clDnnHandle_)->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                  sizeof(size_t), &maxWorkGroupSize, NULL);
  std::cout << "Max work group size : " << maxWorkGroupSize << std::endl;

  cl_ulong localMemSize = 0;
  clGetDeviceInfo((clDnnHandle_)->device, CL_DEVICE_LOCAL_MEM_SIZE,
                  sizeof(cl_ulong), &localMemSize, NULL);
  std::cout << "local Mem size : " << localMemSize << std::endl;
}

cl_int Device::clDnnCreate(clDnnHandle_t **handle) {
  cl_int err = 0;

  *handle = new clDnnHandle_t;
  CheckCl(clGetPlatformIDs(1, &((*handle)->platform), NULL));
  CheckCl(clGetDeviceIDs((*handle)->platform, CL_DEVICE_TYPE_GPU, 1,
                         &((*handle)->device), NULL));
  (*handle)->context =
      clCreateContext(NULL, 1, &((*handle)->device), NULL, NULL, &err);
  CheckCl(err);
  (*handle)->queue = clCreateCommandQueue((*handle)->context, (*handle)->device,
                                          CL_QUEUE_PROFILING_ENABLE, &err);
  CheckCl(err);

  std::string sourceStr;

  err = convertToString("../../src/kernels/kernels.cl", sourceStr);
  CheckCl(err);

  const char *source = sourceStr.c_str();
  size_t sourceSize = sourceStr.length();

  (*handle)->program = clCreateProgramWithSource((*handle)->context, 1, &source,
                                                 &sourceSize, &err);
  CheckCl(err);

  err = clBuildProgram((*handle)->program, 1, &(*handle)->device,
                       "-cl-std=CL2.0", NULL, NULL);

  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo((*handle)->program, (*handle)->device,
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo((*handle)->program, (*handle)->device,
                          CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("%s\n", log);
    delete log;
  }
  CheckCl(err);

  CheckBuildProgram((*handle)->device, (*handle)->program, err);
  CheckCl(err);

  size_t nameSize = 0;
  clGetPlatformInfo((*handle)->platform, CL_PLATFORM_NAME, 0, NULL, &nameSize);
  char *name = (char *)malloc(nameSize);
  clGetPlatformInfo((*handle)->platform, CL_PLATFORM_NAME, nameSize, name, 0);

  std::cout << "Platform name : " << name << std::endl;

  free(name);
  return err;
}

cl_int Device::clDnnDestroy(clDnnHandle_t *handle) {
  cl_int err = 0;
  CheckCl(clReleaseCommandQueue(handle->queue));
  CheckCl(clReleaseContext(handle->context));
  CheckCl(clReleaseDevice(handle->device));

  delete handle;
  return err;
}

Device::~Device() { CheckCl(clDnnDestroy(clDnnHandle_)); }

void Device::GetLocalSize(size_t *localSize, int ndim) {
  for (int i = 0; i < ndim; ++i) {
    localSize[i] = workGroupSize[i];
  }
}

void Device::Flush() {
  cl_int err = clFinish(clDnnHandle_->queue);
  if (err != 0) {
    std::cout << "Flush error" << std::endl;
  }
}

clDnnHandle_t *Device::ClDnnHandle() const { return clDnnHandle_; }
}  // namespace gpu
