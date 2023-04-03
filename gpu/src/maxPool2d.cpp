#include <maxPool2d.h>
#include <util.h>

#include <iostream>

namespace gpu {
namespace op {

MaxPool2d::MaxPool2d(gpu::Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, int kernelSize, int stride)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride) {
  outputChannel_ = inputChannel_;
  outputHeight_ = (inputHeight_ - kernelSize_) / stride_ + 1;
  outputWidth_ = (inputWidth_ - kernelSize) / stride_ + 1;

  inputImageSize = inputChannel_ * inputHeight_ * inputWidth_;

  cl_int err = 0;
  kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "maxPool2d", &err);
  CheckCl(err);
  err = 0;
  kernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "maxPool2d_img", &err);
  CheckCl(err);
}

MaxPool2d::~MaxPool2d() {}

void MaxPool2d::Forward(gpu::Memory *input, gpu::Memory *output) {
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
    err = clSetKernelArg(kernel_, 8, sizeof(cl_int), &kernelSize_);
    err = clSetKernelArg(kernel_, 9, sizeof(cl_int), &stride_);
    CheckCl(err);

    size_t globalSize[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                            (size_t)outputChannel_};
    // size_t localSize[3] = {64, 4, 4};
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
    inputImage = input->createImage2(inputImageSize);

    err = clSetKernelArg(kernel_img, 0, sizeof(cl_mem), &inputImage);
    err = clSetKernelArg(kernel_img, 1, sizeof(cl_mem), &outputMem);
    err = clSetKernelArg(kernel_img, 2, sizeof(cl_int), &inputChannel_);
    err = clSetKernelArg(kernel_img, 3, sizeof(cl_int), &inputHeight_);
    err = clSetKernelArg(kernel_img, 4, sizeof(cl_int), &inputWidth_);
    err = clSetKernelArg(kernel_img, 5, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_img, 6, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_img, 7, sizeof(cl_int), &outputWidth_);
    err = clSetKernelArg(kernel_img, 8, sizeof(cl_int), &kernelSize_);
    err = clSetKernelArg(kernel_img, 9, sizeof(cl_int), &stride_);
    CheckCl(err);

    size_t globalSize[3] = {(size_t)outputWidth_, (size_t)outputHeight_,
                            (size_t)outputChannel_};
    // size_t localSize[3] = {64, 4, 4};
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
