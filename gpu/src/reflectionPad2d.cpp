#include "reflectionPad2d.h"

#include <util.h>

namespace gpu {
namespace op {

ReflectionPad2d::ReflectionPad2d(Device *dev, int inputChannel, int inputHeight,
                                 int inputWidth, std::vector<int> paddings)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      paddings_(paddings) {
  // padding range check
  if (paddings[0] >= inputWidth_ || paddings[1] >= inputWidth_ ||
      paddings[2] >= inputHeight_ || paddings[3] >= inputHeight_) {
    std::cerr
        << "Padding size should be less than the corresponding input dimension"
        << std::endl;
    abort();
  }

  outputChannel_ = inputChannel_;
  outputHeight_ = inputHeight_ + paddings_[2] +
                  paddings_[3];  // height = heitht + topPad + bottomPad
  outputWidth_ = inputWidth_ + paddings_[0] +
                 paddings_[1];  // width = width + leftPad + rightPad

  inputImageSize = inputChannel_ * inputHeight_ * inputWidth_;

  cl_int err = 0;
  kernel_ =
      clCreateKernel(dev_->ClDnnHandle()->program, "reflectionPad2d", &err);
  CheckCl(err);
  err = 0;
  kernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "reflectionPad2d_img", &err);
  CheckCl(err);
}

ReflectionPad2d::ReflectionPad2d(Device *dev, int inputChannel, int inputHeight,
                                 int inputWidth, int paddings)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      singlePadding_(paddings) {
  paddings_ = {singlePadding_, singlePadding_, singlePadding_, singlePadding_};
  // padding range check
  if (paddings_[0] >= inputWidth_ || paddings_[1] >= inputWidth_ ||
      paddings_[2] >= inputHeight_ || paddings_[3] >= inputHeight_) {
    std::cerr
        << "Padding size should be less than the corresponding input dimension"
        << std::endl;
    abort();
  }

  outputChannel_ = inputChannel_;
  outputHeight_ = inputHeight_ + paddings_[2] +
                  paddings_[3];  // height = heitht + topPad + bottomPad
  outputWidth_ = inputWidth_ + paddings_[0] +
                 paddings_[1];  // width = width + leftPad + rightPad

  inputImageSize = inputChannel_ * inputHeight_ * inputWidth_;

  cl_int err = 0;
  kernel_ =
      clCreateKernel(dev_->ClDnnHandle()->program, "reflectionPad2d", &err);
  CheckCl(err);
  err = 0;
  kernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "reflectionPad2d_img", &err);
  CheckCl(err);
}

ReflectionPad2d::~ReflectionPad2d() {}

void ReflectionPad2d::Forward(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();
  cl_mem inputImage;
  cl_int err = 0;

  if (inputImageSize > 134217728 * 2) {
    // if(true){
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputMem);
    err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &outputMem);
    err = clSetKernelArg(kernel_, 2, sizeof(cl_int), &inputChannel_);
    err = clSetKernelArg(kernel_, 3, sizeof(cl_int), &inputHeight_);
    err = clSetKernelArg(kernel_, 4, sizeof(cl_int), &inputWidth_);
    err = clSetKernelArg(kernel_, 5, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_, 6, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_, 7, sizeof(cl_int), &outputWidth_);
    err = clSetKernelArg(kernel_, 8, sizeof(cl_int), &paddings_[0]);
    err = clSetKernelArg(kernel_, 9, sizeof(cl_int), &paddings_[1]);
    err = clSetKernelArg(kernel_, 10, sizeof(cl_int), &paddings_[2]);
    err = clSetKernelArg(kernel_, 11, sizeof(cl_int), &paddings_[3]);
    CheckCl(err);

    size_t globalSize[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                            (size_t)outputChannel_};
    size_t localSize[3] = {16, 16, 1};

    globalSize[0] =
        (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];
    globalSize[1] =
        (globalSize[1] + localSize[1] - 1) / localSize[1] * localSize[1];
    globalSize[2] =
        (globalSize[2] + localSize[2] - 1) / localSize[2] * localSize[2];

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_, 3, NULL,
                                   globalSize, localSize, 0, NULL, NULL));
  } else {
    inputImage = input->createImage2(inputImageSize);

    err = clSetKernelArg(kernel_img, 0, sizeof(cl_mem), &inputImage);
    err = clSetKernelArg(kernel_img, 1, sizeof(cl_mem), &outputMem);
    err = clSetKernelArg(kernel_img, 2, sizeof(cl_int), &inputChannel_);
    err = clSetKernelArg(kernel_img, 3, sizeof(cl_int), &inputHeight_);
    err = clSetKernelArg(kernel_img, 4, sizeof(cl_int), &inputWidth_);
    err = clSetKernelArg(kernel_img, 5, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_img, 6, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_img, 7, sizeof(cl_int), &outputWidth_);
    err = clSetKernelArg(kernel_img, 8, sizeof(cl_int), &paddings_[0]);
    err = clSetKernelArg(kernel_img, 9, sizeof(cl_int), &paddings_[1]);
    err = clSetKernelArg(kernel_img, 10, sizeof(cl_int), &paddings_[2]);
    err = clSetKernelArg(kernel_img, 11, sizeof(cl_int), &paddings_[3]);
    CheckCl(err);

    size_t globalSize[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                            (size_t)outputChannel_};
    size_t localSize[3] = {16, 16, 1};

    globalSize[0] =
        (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];
    globalSize[1] =
        (globalSize[1] + localSize[1] - 1) / localSize[1] * localSize[1];
    globalSize[2] =
        (globalSize[2] + localSize[2] - 1) / localSize[2] * localSize[2];

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_img, 3,
                                   NULL, globalSize, localSize, 0, NULL, NULL));
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}
}  // namespace op
}  // namespace gpu
