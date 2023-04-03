#include "convolution2d.h"

#include <immintrin.h>
#include <omp.h>
#include <util.h>

namespace cpu {
namespace op {

Convolution2d::Convolution2d(cpu::Device *dev, int inputChannel,
                             int inputHeight, int inputWidth, int outputChannel,
                             int kernelSize, int stride)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride) {
  outputChannel_ = outputChannel;
  outputHeight_ = (inputHeight_ - kernelSize_) / stride_ + 1;
  outputWidth_ = (inputWidth_ - kernelSize_) / stride_ + 1;

  weightSize_ =
      outputChannel_ * inputChannel_ * kernelSize * kernelSize * sizeof(float);
  weight_ = new cpu::Memory(dev, weightSize_);

  biasSize_ = outputChannel_ * sizeof(float);
  bias_ = new cpu::Memory(dev, biasSize_);
}

void Convolution2d::LoadWeight(std::ifstream &fp) {
  char *weight_H = (char *)malloc(weightSize_);
  fp.read(weight_H, weightSize_);
  weight_->Set(weight_H);
  delete weight_H;

  char *bias_H = (char *)malloc(biasSize_);
  fp.read(bias_H, biasSize_);
  bias_->Set(bias_H);
  delete bias_H;
}

void Convolution2d::Forward(cpu::Memory *input, cpu::Memory *output) {
  // input shape : (inputChannel_, inputHeight_, inputWidth_)
  // output shape : (outputChannel_, inputHeight_, inputWidth_)
  // weight shape : (outputChannel_, inputChannel_, kernelSize_, kernelSize_)
  // bias shape : (outputChannel)

  float *I = reinterpret_cast<float *>(input->Ptr());
  float *O = reinterpret_cast<float *>(output->Ptr());
  float *W = reinterpret_cast<float *>(weight_->Ptr());
  float *B = reinterpret_cast<float *>(bias_->Ptr());

#pragma omp parallel for
  for (int k = 0; k < outputChannel_; ++k) {
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        float sum = 0.0f;
        for (int c = 0; c < inputChannel_; ++c) {
          for (int r = 0; r < kernelSize_; ++r) {
            for (int s = 0; s < kernelSize_; ++s) {
              int ih = oh * stride_ + r;
              int iw = ow * stride_ + s;
              if (0 <= ih && ih < inputHeight_ && 0 <= iw && iw < inputWidth_)
                sum += I[(c * inputHeight_ + ih) * inputWidth_ + iw] *
                       W[((k * inputChannel_ + c) * kernelSize_ + r) *
                             kernelSize_ +
                         s];
            }
          }
        }
        // bias
        sum += B[k];
        // O[k][oh][ow] = sum;
        O[(k * outputHeight_ + oh) * outputWidth_ + ow] = sum;
      }
    }
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

void Convolution2d::PrintWeight() {
  std::cout << "weight : " << std::endl;
  float *W = reinterpret_cast<float *>(weight_->Ptr());
  for (int k = 0; k < outputChannel_; ++k) {
    for (int c = 0; c < inputChannel_; ++c) {
      for (int r = 0; r < kernelSize_; ++r) {
        for (int s = 0; s < kernelSize_; ++s) {
          int idx =
              ((k * inputChannel_ + c) * kernelSize_ + r) * kernelSize_ + s;
          std::cout << W[idx] << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "bias : " << std::endl;
  float *B = reinterpret_cast<float *>(bias_->Ptr());
  for (int k = 0; k < outputChannel_; ++k) {
    std::cout << B[k] << ", ";
  }
  std::cout << std::endl;
}

Convolution2d::~Convolution2d() {}

}  // namespace op
}  // namespace cpu
