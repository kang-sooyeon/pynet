#include "upsample.h"

#include "util.h"

namespace gpu {
namespace op {

Upsample::Upsample(Device *dev, int inputChannel, int inputHeight,
                   int inputWidth, int scale)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      scale_(scale) {
  outputChannel_ = inputChannel_;
  outputHeight_ = inputHeight_ * scale_;
  outputWidth_ = inputWidth_ * scale_;

  cl_int err = 0;
  kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "upsample_", &err);
  CheckCl(err);
}

Upsample::~Upsample() {}

void Upsample::Forward(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();
  cl_int err = 0;

  err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputMem);
  err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &outputMem);
  err = clSetKernelArg(kernel_, 2, sizeof(cl_int), &inputChannel_);
  err = clSetKernelArg(kernel_, 3, sizeof(cl_int), &inputHeight_);
  err = clSetKernelArg(kernel_, 4, sizeof(cl_int), &inputWidth_);
  err = clSetKernelArg(kernel_, 5, sizeof(cl_int), &outputChannel_);
  err = clSetKernelArg(kernel_, 6, sizeof(cl_int), &outputHeight_);
  err = clSetKernelArg(kernel_, 7, sizeof(cl_int), &outputWidth_);
  err = clSetKernelArg(kernel_, 8, sizeof(cl_int), &scale_);
  CheckCl(err);

  size_t globalSize[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                          (size_t)outputChannel_};
  size_t localSize[3] = {16, 4, 4};

  globalSize[0] =
      (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];
  globalSize[1] =
      (globalSize[1] + localSize[1] - 1) / localSize[1] * localSize[1];
  globalSize[2] =
      (globalSize[2] + localSize[2] - 1) / localSize[2] * localSize[2];

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_, 3, NULL,
                                 globalSize, localSize, 0, NULL, NULL));

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

}  // namespace op
}  // namespace gpu
