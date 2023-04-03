#include "sigmoid.h"

#include "util.h"

namespace gpu {
namespace op {

Sigmoid::Sigmoid(Device *dev, int inputChannel, int inputHeight, int inputWidth)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth) {
  outputChannel_ = inputChannel;
  outputHeight_ = inputHeight;
  outputWidth_ = inputWidth;

  cl_int err = 0;
  kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "sigmoid", &err);
  CheckCl(err);
  err = 0;
  kernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "sigmoid_img", &err);
  CheckCl(err);
}

Sigmoid::~Sigmoid() {}

void Sigmoid::Forward(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();

  cl_mem inputImage;
  size_t inSize = inputChannel_ * inputHeight_ * inputWidth_;

  cl_int err = 0;

  if (inSize > 134217728 * 2) {
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputMem);
    err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &outputMem);
    err = clSetKernelArg(kernel_, 2, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_, 3, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_, 4, sizeof(cl_int), &outputWidth_);
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
  } else {
    inputImage = input->createImage2(inSize);
    outputWidth_ /= 2;
    err = clSetKernelArg(kernel_img, 0, sizeof(cl_mem), &inputImage);
    err = clSetKernelArg(kernel_img, 1, sizeof(cl_mem), &outputMem);
    err = clSetKernelArg(kernel_img, 2, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_img, 3, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_img, 4, sizeof(cl_int), &outputWidth_);
    CheckCl(err);
    outputWidth_ *= 2;
    size_t globalSize[3] = {(size_t)outputWidth_ / 2, (size_t)outputHeight_,
                            (size_t)outputChannel_};
    size_t localSize[3] = {16, 4, 4};

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
