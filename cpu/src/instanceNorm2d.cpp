#include "instanceNorm2d.h"

#include <omp.h>

#define EPSILON 0.00001
#define SQUARE(x) ((x) * (x))

namespace cpu {
namespace op {

InstanceNorm2d::InstanceNorm2d(Device *dev, int inputChannel, int inputHeight,
                               int inputWidth, bool affine)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      affine_(affine) {
  eps_ = EPSILON;
  momentum_ = 0.1;
  trackRunningStats_ = false;
  outputChannel_ = inputChannel;
  outputHeight_ = inputHeight;
  outputWidth_ = inputWidth;

  weightSize_ = outputChannel_ * sizeof(float);
  weight_ = new cpu::Memory(dev, weightSize_);

  biasSize_ = outputChannel_ * sizeof(float);
  bias_ = new cpu::Memory(dev, biasSize_);
}

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

InstanceNorm2d::~InstanceNorm2d() {}

void InstanceNorm2d::Forward(cpu::Memory *input) {
  float *I = reinterpret_cast<float *>(input->Ptr());
  float *W = reinterpret_cast<float *>(weight_->Ptr());
  float *B = reinterpret_cast<float *>(bias_->Ptr());

  int size = inputHeight_ * inputWidth_;

#pragma omp parallel for
  for (int k = 0; k < outputChannel_; ++k) {
    // mean
    float mean = 0.0f;
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        mean += I[(k * inputHeight_ + oh) * inputWidth_ + ow];
      }
    }
    mean /= size;

    // standard variance
    float std = 0.0f, x;
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        x = I[(k * inputHeight_ + oh) * inputWidth_ + ow] - mean;
        std += x * x;
      }
    }
    std /= size;
    std += EPSILON;
    std = sqrt(std);

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        int idx = (k * inputHeight_ + oh) * inputWidth_ + ow;
        I[idx] = W[k] * ((I[idx] - mean) / std) + B[k];
      }
    }
  }

  input->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                     sizeof(float));
}

void InstanceNorm2d::PrintWeight() {
  std::cout << "weight : " << std::endl;
  float *W = reinterpret_cast<float *>(weight_->Ptr());
  for (int k = 0; k < outputChannel_; ++k) {
    std::cout << W[k] << ", ";
  }
  std::cout << std::endl;

  std::cout << "bias : " << std::endl;
  float *B = reinterpret_cast<float *>(bias_->Ptr());
  for (int k = 0; k < outputChannel_; ++k) {
    std::cout << B[k] << ", ";
  }
  std::cout << std::endl;
}

}  // namespace op
}  // namespace cpu
