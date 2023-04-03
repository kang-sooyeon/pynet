#include "mseLoss.h"

#include <util.h>

namespace gpu {
namespace op {

MSELoss::MSELoss(Device *dev, int inputChannel, int inputHeight, int inputWidth,
                 int reduction)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      reduction_(reduction) {
  // reduction
  // 0 = mean
  // 1 = sum
  outputChannel_ = 1;
  outputHeight_ = 1;
  outputWidth_ = 1;

  size_ = inputChannel * inputHeight * inputWidth;

  cl_int err = 0;
  if (reduction_ == 0)
    kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "mseLossMean", &err);
  else if (reduction_ == 1)
    kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "mseLossSum", &err);
  else {
    cerr << "Wrong reduction options in MSELoss !!\n";
    abort();
  }
  CheckCl(err);
}

MSELoss::~MSELoss() {}

void MSELoss::Forward(gpu::Memory *input1, gpu::Memory *input2,
                      gpu::Memory *output) {
  cl_mem inputMem1 = input1->Ptr();
  cl_mem inputMem2 = input2->Ptr();
  cl_mem outputMem = output->Ptr();
  cl_int err = 0;

  size_t globalSize[3] = {(size_t)inputWidth_, (size_t)inputHeight_,
                          (size_t)inputChannel_};
  size_t localSize[3] = {16, 16, 4};

  globalSize[0] =
      (globalSize[0] + localSize[0] - 1) / localSize[0] * localSize[0];
  globalSize[1] =
      (globalSize[1] + localSize[1] - 1) / localSize[1] * localSize[1];
  globalSize[2] =
      (globalSize[2] + localSize[2] - 1) / localSize[2] * localSize[2];

  int tmpSize = (globalSize[0] / localSize[0]) *
                (globalSize[1] / localSize[1]) * (globalSize[2] / localSize[2]);

  cl_mem tmp = clCreateBuffer(dev_->ClDnnHandle()->context, CL_MEM_READ_WRITE,
                              tmpSize * sizeof(DATA_TYPE), NULL, &err);
  CheckCl(err);

  err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputMem1);
  err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &inputMem2);
  err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &tmp);
  err = clSetKernelArg(kernel_, 3, sizeof(cl_mem), &outputMem);

  err = clSetKernelArg(kernel_, 4, sizeof(cl_int), &inputChannel_);
  err = clSetKernelArg(kernel_, 5, sizeof(cl_int), &inputHeight_);
  err = clSetKernelArg(kernel_, 6, sizeof(cl_int), &inputWidth_);

  err = clSetKernelArg(kernel_, 7, sizeof(cl_int), &tmpSize);

  CheckCl(err);

  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_, 3, NULL,
                                 globalSize, localSize, 0, NULL, NULL));
  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

}  // namespace op
}  // namespace gpu
