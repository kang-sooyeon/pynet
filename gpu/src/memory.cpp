#include <memory.h>
#include <util.h>

#include <cstring>

#include "half.hpp"

using namespace std;
using half_float::half;
using namespace half_float::literal;

namespace gpu {
size_t Memory::allocSizeAll_ = 0;

size_t Memory::AllocSizeAll() { return allocSizeAll_; }

void Memory::SetAllocSizeAll(size_t v) { allocSizeAll_ = v; }

Memory::Memory(Device *dev) { dev_ = dev; }

Memory::Memory(Device *dev, size_t size)
    : allocSize_(size), usedSize_(size), dev_(dev) {
  cl_int err = 0;
  ptr_ = clCreateBuffer(dev_->ClDnnHandle()->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL,
                        &err);
  CheckCl(err);
  allocSizeAll_ += allocSize_;
}

cl_mem Memory::createImage(int inputImageSize) {

  cl_image_format img_format = {CL_R, CL_HALF_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE1D_BUFFER,
                              size_t(inputImageSize),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              ptr_};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  images_.push_back(imagePtr);

  return imagePtr;
}

cl_mem Memory::createImage2D(int inputImageSizeX, int inputImageSizeY) {

  cl_image_format img_format = {CL_R, CL_HALF_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE2D,
                              size_t(inputImageSizeX),
                              size_t(inputImageSizeY),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              ptr_};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  images_.push_back(imagePtr);

  return imagePtr;
}

cl_mem Memory::createImage2D(cl_mem buf, int inputImageSizeX,
                             int inputImageSizeY) {

  cl_image_format img_format = {CL_R, CL_HALF_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE2D,
                              size_t(inputImageSizeX),
                              size_t(inputImageSizeY),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              buf};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  return imagePtr;
}

cl_mem Memory::createImageFloat2(cl_mem buf, int inputImageSize) {
  cl_image_format img_format = {CL_RG, CL_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE1D_BUFFER,
                              size_t(inputImageSize / 2),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              buf};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  return imagePtr;
}

cl_mem Memory::createImageFloat2(int inputImageSize) {
  cl_image_format img_format = {CL_RG, CL_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE1D_BUFFER,
                              size_t(inputImageSize / 2),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              ptr_};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  images_.push_back(imagePtr);

  return imagePtr;
}

cl_mem Memory::createImage2(cl_mem buf, int inputImageSize) {

  cl_image_format img_format = {CL_RG, CL_HALF_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE1D_BUFFER,
                              size_t(inputImageSize / 2),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              buf};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  return imagePtr;
}

cl_mem Memory::createImage2(int inputImageSize) {

  cl_image_format img_format = {CL_RG, CL_HALF_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE1D_BUFFER,
                              size_t(inputImageSize / 2),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              ptr_};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  images_.push_back(imagePtr);

  return imagePtr;
}

cl_mem Memory::createImage(cl_mem buf, size_t imageSize) {

  cl_image_format img_format = {CL_R, CL_HALF_FLOAT};

  cl_image_desc input_desc = {CL_MEM_OBJECT_IMAGE1D_BUFFER,
                              size_t(imageSize),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              buf};

  cl_int err = 0;
  cl_mem imagePtr =
      clCreateImage(dev_->ClDnnHandle()->context, CL_MEM_READ_ONLY, &img_format,
                    &input_desc, NULL, &err);
  CheckCl(err);

  return imagePtr;
}

void Memory::addImage(cl_mem usedImage) { images_.push_back(usedImage); }

void Memory::releaseImages() {
  for (int i = 0; i < images_.size(); ++i)
    CheckCl(clReleaseMemObject(images_[i]));

  images_.clear();
}

void Memory::CreateBuffer(size_t size, bool is_alloc_host_ptr) {
  usedSize_ = size;
  allocSize_ = size;

  cl_int err = 0;

  if (is_alloc_host_ptr)
    ptr_ = clCreateBuffer(dev_->ClDnnHandle()->context,
                          CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL,
                          &err);
  // else
  // ptr_ = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE |
  // CL_MEM_USE_HOST_PTR, size, NULL, &err); ptr_ =
  // clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE |
  // CL_MEM_COPY_HOST_PTR, size, NULL, &err);

  CheckCl(err);
  allocSizeAll_ += allocSize_;
}

void Memory::SetUsedSize(size_t usedSize) { usedSize_ = usedSize; }

void Memory::Release() {
  releaseImages();
  allocSizeAll_ -= allocSize_;
  CheckCl(clReleaseMemObject(ptr_));
}

Memory::~Memory() {
  allocSizeAll_ -= allocSize_;
  CheckCl(clReleaseMemObject(ptr_));
}

cl_mem Memory::Ptr() { return ptr_; }

size_t Memory::Size() { return usedSize_; }

void Memory::Set(void *src) {
  CheckCl(clEnqueueWriteBuffer(dev_->ClDnnHandle()->queue, ptr_, CL_TRUE, 0,
                               usedSize_, src, 0, NULL, NULL));
}

void Memory::Get(void *dst) {
  CheckCl(clEnqueueReadBuffer(dev_->ClDnnHandle()->queue, ptr_, CL_TRUE, 0,
                              usedSize_, dst, 0, NULL, NULL));
}

Device *Memory::Dev() { return dev_; }

void Memory::Print() {
  void *dst = malloc(usedSize_);
  Get(dst);
  DATA_TYPE *O = reinterpret_cast<DATA_TYPE *>(dst);

  int flag = 2;

  switch (flag) {
    case 1:
      std::cout << std::endl;
      for (size_t i = 0; i < usedSize_ / sizeof(DATA_TYPE); i++) {
        std::cout << (float)(O[i]) << ", ";
      }
      std::cout << std::endl;
      break;

    case 2:
      for (size_t i = 0; i < 4; i++) {
        std::cout << (float)(O[i]) << ", ";
      }
      std::cout << std::endl;
      break;
  }
  delete dst;
}

void Memory::Print(int n) {
  printf("usedSize : %d\n", usedSize_);
  void *dst = malloc(usedSize_);
  Get(dst);
  float *O = reinterpret_cast<float *>(dst);

  int flag = 3;

  switch (flag) {
    case 1:
      std::cout << std::endl;
      for (size_t i = 0; i < usedSize_ / sizeof(float); i++) {
        std::cout << O[i] << ", ";
      }
      std::cout << std::endl;
      break;

    case 2:
      for (size_t i = 0; i < n; i++) {
        std::cout << O[i] << ", ";
      }

      std::cout << std::endl;
      break;

    case 3:
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          std::cout << O[i * n + j] << ", ";
        }
        std::cout << std::endl;
      }

      break;
  }
  delete dst;
}

size_t Memory::Sumt() {
  void *dst = malloc(usedSize_);
  Get(dst);
  size_t *O = reinterpret_cast<size_t *>(dst);

  size_t sum = 0;
  for (size_t i = 0; i < usedSize_ / sizeof(size_t); i++) {
    sum += O[i];
  }
  delete dst;
  return sum;
}

double Memory::Sumf() {
  // void* dst = malloc(usedSize_);
  char *dst = (char *)malloc(usedSize_);
  Get(dst);
  DATA_TYPE *O = reinterpret_cast<DATA_TYPE *>(dst);

  double sum = 0;
  for (size_t i = 0; i < usedSize_ / sizeof(DATA_TYPE); i++) {
    sum += (float)(O[i]);
  }
  delete dst;
  return sum;
}

}  // namespace gpu
