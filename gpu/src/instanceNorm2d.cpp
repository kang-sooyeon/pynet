#include "instanceNorm2d.h"

#include <util.h>

#define EPSILON 0.00001

namespace gpu {
namespace op {

InstanceNorm2d::InstanceNorm2d(Device *dev, int inputChannel, int inputHeight,
                               int inputWidth, bool affine, bool reluOn)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      reluOn_(reluOn) {
  outputChannel_ = inputChannel;
  outputHeight_ = inputHeight;
  outputWidth_ = inputWidth;

  weightSize_ = outputChannel_ * sizeof(DATA_TYPE);
  weight_ = new gpu::Memory(dev, weightSize_);

  biasSize_ = outputChannel_ * sizeof(DATA_TYPE);
  bias_ = new gpu::Memory(dev, biasSize_);

  cl_int err = 0;
  reductionKernel_ =
      clCreateKernel(dev_->ClDnnHandle()->program, "reduction", &err);
  CheckCl(err);
  reductionKernel2_ =
      clCreateKernel(dev_->ClDnnHandle()->program, "reduction_float", &err);
  CheckCl(err);
  meanKernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "mean", &err);
  CheckCl(err);
  subSquareKernel_ =
      clCreateKernel(dev_->ClDnnHandle()->program, "subSquare", &err);
  CheckCl(err);
  stdKernel_ = clCreateKernel(dev_->ClDnnHandle()->program, "std", &err);
  CheckCl(err);
  instanceKernel_ =
      clCreateKernel(dev_->ClDnnHandle()->program, "instanceNorm2d", &err);
  CheckCl(err);

  reductionKernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "reduction_img", &err);
  CheckCl(err);
  reductionKernel2_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "reduction_float_img", &err);
  CheckCl(err);
  meanKernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "mean_img", &err);
  CheckCl(err);
  subSquareKernel_img =
      clCreateKernel(dev_->ClDnnHandle()->program, "subSquare_img", &err);
  CheckCl(err);
  stdKernel_img = clCreateKernel(dev_->ClDnnHandle()->program, "std_img", &err);
  CheckCl(err);
}

gpu::Memory *InstanceNorm2d::getWeights() { return weight_; }

gpu::Memory *InstanceNorm2d::getBias() { return bias_; }

void InstanceNorm2d::LoadWeight(std::ifstream &fp) {
  char *weight_H = (char *)malloc(weightSize_);
  fp.read(weight_H, weightSize_);
  weight_->Set(weight_H);
  delete weight_H;

  char *bias_H = (char *)malloc(biasSize_);
  fp.read(bias_H, biasSize_);
  bias_->Set(bias_H);
  delete bias_H;
}

InstanceNorm2d::~InstanceNorm2d() {
  delete weight_;
  delete bias_;
}

void InstanceNorm2d::Forward(gpu::Memory *output, gpu::Memory *tmp) {
  ////////////////////////////////////// reduction kernel
  cl_mem outputMem = output->Ptr();
  cl_mem tmpMem = tmp->Ptr();
  cl_mem weightMem = weight_->Ptr();
  cl_mem biasMem = bias_->Ptr();

  cl_mem outputImage, tmpImage, weightImage, biasImage;

  int inputImaegSize = inputChannel_ * inputHeight_ * inputWidth_;
  int HW = outputHeight_ * outputWidth_;
  int localNum = 256;
  int groupNum = (HW + localNum - 1) / localNum;

  cl_int err = 0;

  if (inputImaegSize > 134217728 * 2) {
    // if(true){
    err = 0;
    clSetKernelArg(reductionKernel_, 0, sizeof(cl_mem), &outputMem);
    clSetKernelArg(reductionKernel_, 1, sizeof(cl_mem), &groupSumMem);
    clSetKernelArg(reductionKernel_, 2, sizeof(float) * localNum, NULL);
    clSetKernelArg(reductionKernel_, 3, sizeof(cl_int), &HW);
    clSetKernelArg(reductionKernel_, 4, sizeof(cl_int), &groupNum);
    clSetKernelArg(reductionKernel_, 5, sizeof(cl_int), &outputChannel_);

    gws[0] = HW;
    lws[0] = localNum;
    gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, reductionKernel_,
                                   1, NULL, gws, lws, 0, NULL, NULL));
  } else {
    outputImage = output->createImage2(inputImaegSize);
    clSetKernelArg(reductionKernel_img, 0, sizeof(cl_mem), &outputImage);
    clSetKernelArg(reductionKernel_img, 1, sizeof(cl_mem), &groupSumMem);
    clSetKernelArg(reductionKernel_img, 2, sizeof(float) * localNum, NULL);
    clSetKernelArg(reductionKernel_img, 3, sizeof(cl_int), &HW);
    clSetKernelArg(reductionKernel_img, 4, sizeof(cl_int), &groupNum);
    clSetKernelArg(reductionKernel_img, 5, sizeof(cl_int), &outputChannel_);

    gws[0] = HW;
    lws[0] = localNum;
    gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                   reductionKernel_img, 1, NULL, gws, lws, 0,
                                   NULL, NULL));
  }

  //////////////////////////////////////////////// mean
  int elems = groupNum * outputChannel_;

  if (elems > 134217728 * 2) {
    clSetKernelArg(meanKernel_, 0, sizeof(cl_mem), &groupSumMem);
    clSetKernelArg(meanKernel_, 1, sizeof(cl_int), &elems);
    clSetKernelArg(meanKernel_, 2, sizeof(cl_int), &groupNum);
    clSetKernelArg(meanKernel_, 3, sizeof(cl_int), &HW);
    clSetKernelArg(meanKernel_, 4, sizeof(cl_mem), &meanMem);

    gws[0] = elems;
    lws[0] = 64;

    gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, meanKernel_, 1,
                                   NULL, gws, lws, 0, NULL, NULL));
  } else {
    cl_mem groupSumMemImage = output->createImageFloat2(groupSumMem, elems);

    clSetKernelArg(meanKernel_img, 0, sizeof(cl_mem), &groupSumMemImage);
    clSetKernelArg(meanKernel_img, 1, sizeof(cl_int), &elems);
    clSetKernelArg(meanKernel_img, 2, sizeof(cl_int), &groupNum);
    clSetKernelArg(meanKernel_img, 3, sizeof(cl_int), &HW);
    clSetKernelArg(meanKernel_img, 4, sizeof(cl_mem), &meanMem);

    gws[0] = elems;
    lws[0] = 64;

    gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, meanKernel_img,
                                   1, NULL, gws, lws, 0, NULL, NULL));
  }

  /////////////////////////////////// sub square
  int offsetStride = outputChannel_ / 2;
  for (int i = 0; i < outputChannel_; i += offsetStride) {
    elems = groupNum * offsetStride;

    if (inputImaegSize > 134217728 * 2) {
      clSetKernelArg(subSquareKernel_, 0, sizeof(cl_mem), &outputMem);
      clSetKernelArg(subSquareKernel_, 1, sizeof(cl_mem), &tmpMem);
      clSetKernelArg(subSquareKernel_, 2, sizeof(cl_mem), &meanMem);
      clSetKernelArg(subSquareKernel_, 3, sizeof(cl_int), &(offsetStride));
      clSetKernelArg(subSquareKernel_, 4, sizeof(cl_int), &outputHeight_);
      clSetKernelArg(subSquareKernel_, 5, sizeof(cl_int), &outputWidth_);
      clSetKernelArg(subSquareKernel_, 6, sizeof(cl_int), &i);

      gws[0] = offsetStride * outputHeight_ * outputWidth_;
      lws[0] = localNum;

      gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     subSquareKernel_, 1, NULL, gws, lws, 0,
                                     NULL, NULL));
    } else {
      cl_mem outputImage2 = output->createImage2(inputImaegSize);

      clSetKernelArg(subSquareKernel_img, 0, sizeof(cl_mem), &outputImage2);
      clSetKernelArg(subSquareKernel_img, 1, sizeof(cl_mem), &tmpMem);
      clSetKernelArg(subSquareKernel_img, 2, sizeof(cl_mem), &meanMem);
      clSetKernelArg(subSquareKernel_img, 3, sizeof(cl_int), &(offsetStride));
      clSetKernelArg(subSquareKernel_img, 4, sizeof(cl_int), &outputHeight_);
      clSetKernelArg(subSquareKernel_img, 5, sizeof(cl_int), &outputWidth_);
      clSetKernelArg(subSquareKernel_img, 6, sizeof(cl_int), &i);

      gws[0] = offsetStride * outputHeight_ * outputWidth_;
      lws[0] = localNum;

      gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     subSquareKernel_img, 1, NULL, gws, lws, 0,
                                     NULL, NULL));
    }

    if (inputImaegSize > 134217728 * 2) {
      // if(true){

      ////////////////////////////////////// reduction for std
      err = clSetKernelArg(reductionKernel2_, 0, sizeof(cl_mem), &tmpMem);
      err = clSetKernelArg(reductionKernel2_, 1, sizeof(cl_mem), &groupSumMem);
      err =
          clSetKernelArg(reductionKernel2_, 2, sizeof(float) * localNum, NULL);
      err = clSetKernelArg(reductionKernel2_, 3, sizeof(cl_int), &HW);
      err = clSetKernelArg(reductionKernel2_, 4, sizeof(cl_int), &groupNum);
      err =
          clSetKernelArg(reductionKernel2_, 5, sizeof(cl_int), &(offsetStride));
      err = clSetKernelArg(reductionKernel2_, 6, sizeof(cl_int), &i);

      gws[0] = HW;
      lws[0] = localNum;
      gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     reductionKernel2_, 1, NULL, gws, lws, 0,
                                     NULL, NULL));
    } else {
      ////////////////////////////////////// reduction for std

      cl_mem tmpMemImage = tmp->createImageFloat2(inputImaegSize / 2);

      err =
          clSetKernelArg(reductionKernel2_img, 0, sizeof(cl_mem), &tmpMemImage);
      err =
          clSetKernelArg(reductionKernel2_img, 1, sizeof(cl_mem), &groupSumMem);
      err = clSetKernelArg(reductionKernel2_img, 2, sizeof(float) * localNum,
                           NULL);
      err = clSetKernelArg(reductionKernel2_img, 3, sizeof(cl_int), &HW);
      err = clSetKernelArg(reductionKernel2_img, 4, sizeof(cl_int), &groupNum);
      err = clSetKernelArg(reductionKernel2_img, 5, sizeof(cl_int),
                           &(offsetStride));
      err = clSetKernelArg(reductionKernel2_img, 6, sizeof(cl_int), &i);

      gws[0] = HW;
      lws[0] = localNum;
      gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     reductionKernel2_img, 1, NULL, gws, lws, 0,
                                     NULL, NULL));
    }

    if (elems > 134217728 * 2) {
      //////////////////////////////////////////////// std
      clSetKernelArg(stdKernel_, 0, sizeof(cl_mem), &groupSumMem);
      clSetKernelArg(stdKernel_, 1, sizeof(cl_int), &elems);
      clSetKernelArg(stdKernel_, 2, sizeof(cl_int), &groupNum);
      clSetKernelArg(stdKernel_, 3, sizeof(cl_int), &HW);
      clSetKernelArg(stdKernel_, 4, sizeof(cl_mem), &stdMem);
      clSetKernelArg(stdKernel_, 5, sizeof(cl_int), &i);

      gws[0] = elems;
      lws[0] = 64;

      gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, stdKernel_, 1,
                                     NULL, gws, lws, 0, NULL, NULL));
    } else {
      //////////////////////////////////////////////// std

      cl_mem groupSumMemImage =
          output->createImageFloat2(groupSumMem, elems * 2);

      clSetKernelArg(stdKernel_img, 0, sizeof(cl_mem), &groupSumMemImage);
      clSetKernelArg(stdKernel_img, 1, sizeof(cl_int), &elems);
      clSetKernelArg(stdKernel_img, 2, sizeof(cl_int), &groupNum);
      clSetKernelArg(stdKernel_img, 3, sizeof(cl_int), &HW);
      clSetKernelArg(stdKernel_img, 4, sizeof(cl_mem), &stdMem);
      clSetKernelArg(stdKernel_img, 5, sizeof(cl_int), &i);

      gws[0] = elems;
      lws[0] = 64;

      gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, stdKernel_img,
                                     1, NULL, gws, lws, 0, NULL, NULL));
    }
  }

  //////////////////////////////////////////  instanceNorm
  clSetKernelArg(instanceKernel_, 0, sizeof(cl_mem), &outputMem);
  clSetKernelArg(instanceKernel_, 1, sizeof(cl_mem), &weightMem);
  clSetKernelArg(instanceKernel_, 2, sizeof(cl_mem), &biasMem);
  clSetKernelArg(instanceKernel_, 3, sizeof(cl_mem), &meanMem);
  clSetKernelArg(instanceKernel_, 4, sizeof(cl_mem), &stdMem);
  clSetKernelArg(instanceKernel_, 5, sizeof(cl_int), &outputChannel_);
  clSetKernelArg(instanceKernel_, 6, sizeof(cl_int), &outputHeight_);
  clSetKernelArg(instanceKernel_, 7, sizeof(cl_int), &outputWidth_);
  clSetKernelArg(instanceKernel_, 8, sizeof(cl_int), &reluOn_);

  gws[0] = outputChannel_ * outputHeight_ * outputWidth_;
  lws[0] = 256;

  gws[0] = (gws[0] + lws[0] - 1) / lws[0] * lws[0];
  CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, instanceKernel_, 1,
                                 NULL, gws, lws, 0, NULL, NULL));

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

}  // namespace op
}  // namespace gpu
