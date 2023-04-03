#include "cat.h"

#include <util.h>

namespace gpu {
namespace op {

// Cat
Cat::Cat(gpu::Device *dev, int inputChannel1, int inputHeight1, int inputWidth1,
         int inputChannel2, int inputHeight2, int inputWidth2, int dim)
    : gpu::op::Operation(dev, inputChannel1, inputHeight1, inputWidth1),
      inputChannel2_(inputChannel2),
      inputHeight2_(inputHeight2),
      inputWidth2_(inputWidth2),
      dim_(dim) {
  cl_int err = 0;

  if (dim == 0) {
    // We don't care about that case
    std::cerr << "Invalid Dimension Parameter!" << std::endl;
    abort();
  } else if (dim == 1) {
    if (inputWidth_ != inputWidth2_ || inputHeight_ != inputHeight2_) {
      std::cerr << "Invalid Input Tensor Size <Cat-channel>" << std::endl;
      std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " , "
                << inputWidth_ << std::endl;
      std::cerr << "<" << inputChannel2_ << " , " << inputHeight2_ << " , "
                << inputWidth2_ << std::endl;
      abort();
    }
    inputImageSize1 = inputChannel1 * inputHeight1 * inputWidth1;
    inputImageSize2 = inputChannel2 * inputHeight2 * inputWidth2;

    outputChannel_ = inputChannel_ + inputChannel2_;
    outputHeight_ = inputHeight_;
    outputWidth_ = inputWidth_;
    kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "cat1", &err);
    kernel_img = clCreateKernel(dev_->ClDnnHandle()->program, "cat1_img", &err);
  } else if (dim == 2) {
    if (inputChannel_ != inputChannel2_ || inputWidth_ != inputWidth2_) {
      std::cerr << "Invalid Input Tensor Size <Cat-height>" << std::endl;
      std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " , "
                << inputWidth_ << std::endl;
      std::cerr << "<" << inputChannel2_ << " , " << inputHeight2_ << " , "
                << inputWidth2_ << std::endl;
      abort();
    }

    outputChannel_ = inputChannel_;
    outputHeight_ = inputHeight_ + inputHeight2_;
    outputWidth_ = inputWidth_;
    kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "cat2", &err);
  } else if (dim == 3) {
    if (inputChannel_ != inputChannel2_ || inputHeight_ != inputHeight2_) {
      std::cerr << "Invalid Input Tensor Size <Cat-width>" << std::endl;
      std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " , "
                << inputWidth_ << std::endl;
      std::cerr << "<" << inputChannel2_ << " , " << inputHeight2_ << " , "
                << inputWidth2_ << std::endl;
      abort();
    }

    outputChannel_ = inputChannel_;
    outputHeight_ = inputHeight_;
    outputWidth_ = inputWidth_ + inputWidth2_;
    kernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "cat3", &err);
  } else {
    std::cerr << "Invalid Dimension Parameter!" << std::endl;
    abort();
  }
  CheckCl(err);
}

Cat::~Cat() {
}

void Cat::Forward(gpu::Memory *input1, gpu::Memory *input2,
                  gpu::Memory *output) {
  cl_mem inputMem1 = input1->Ptr();
  cl_mem inputMem2 = input2->Ptr();
  cl_mem outputMem = output->Ptr();
  cl_mem inputImage1, inputImage2;
  cl_int err = 0;

  if (inputImageSize1 > 134217728 * 2 || inputImageSize2 > 134217728 * 2) {
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputMem1);
    err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &inputMem2);
    err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &outputMem);

    err = clSetKernelArg(kernel_, 3, sizeof(cl_int), &inputChannel_);
    err = clSetKernelArg(kernel_, 4, sizeof(cl_int), &inputHeight_);
    err = clSetKernelArg(kernel_, 5, sizeof(cl_int), &inputWidth_);

    err = clSetKernelArg(kernel_, 6, sizeof(cl_int), &inputChannel2_);
    err = clSetKernelArg(kernel_, 7, sizeof(cl_int), &inputHeight2_);
    err = clSetKernelArg(kernel_, 8, sizeof(cl_int), &inputWidth2_);

    err = clSetKernelArg(kernel_, 9, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_, 10, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_, 11, sizeof(cl_int), &outputWidth_);

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
    inputImage1 = input1->createImage2(inputImageSize1);
    inputImage2 = input2->createImage2(inputImageSize2);

    err = clSetKernelArg(kernel_img, 0, sizeof(cl_mem), &inputImage1);
    err = clSetKernelArg(kernel_img, 1, sizeof(cl_mem), &inputImage2);
    err = clSetKernelArg(kernel_img, 2, sizeof(cl_mem), &outputMem);

    err = clSetKernelArg(kernel_img, 3, sizeof(cl_int), &inputChannel_);
    err = clSetKernelArg(kernel_img, 4, sizeof(cl_int), &inputHeight_);
    err = clSetKernelArg(kernel_img, 5, sizeof(cl_int), &inputWidth_);

    err = clSetKernelArg(kernel_img, 6, sizeof(cl_int), &inputChannel2_);
    err = clSetKernelArg(kernel_img, 7, sizeof(cl_int), &inputHeight2_);
    err = clSetKernelArg(kernel_img, 8, sizeof(cl_int), &inputWidth2_);

    err = clSetKernelArg(kernel_img, 9, sizeof(cl_int), &outputChannel_);
    err = clSetKernelArg(kernel_img, 10, sizeof(cl_int), &outputHeight_);
    err = clSetKernelArg(kernel_img, 11, sizeof(cl_int), &outputWidth_);

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
