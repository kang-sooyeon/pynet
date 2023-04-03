#include "maxPool2d.h"

#include <omp.h>

namespace cpu {
namespace op {

MaxPool2d::MaxPool2d(Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, int kernelSize, int stride)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride) {
  outputChannel_ = inputChannel_;
  outputHeight_ = (inputHeight_ - kernelSize_) / stride_ + 1;
  outputWidth_ = (inputWidth_ - kernelSize) / stride_ + 1;
}

MaxPool2d::~MaxPool2d() {}

void MaxPool2d::Forward(cpu::Memory *input, cpu::Memory *output) {
  float *I = reinterpret_cast<float *>(input->Ptr());
  float *O = reinterpret_cast<float *>(output->Ptr());

#pragma omp parallel for
  for (int c = 0; c < outputChannel_; ++c) {
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        float max = -FLT_MAX;
        for (int r = 0; r < kernelSize_; ++r) {
          for (int s = 0; s < kernelSize_; ++s) {
            int ih = stride_ * oh + r;
            int iw = stride_ * ow + s;
            int idx = (c * inputHeight_ + ih) * inputWidth_ + iw;
            float val = I[idx];
            if (max < val) max = val;
          }
        }
        O[(c * outputHeight_ + oh) * outputWidth_ + ow] = max;
      }
    }
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

}  // namespace op
}  // namespace cpu
