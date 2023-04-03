#include "leakyReLU.h"

#include <omp.h>

namespace cpu {
namespace op {

LeakyReLU::LeakyReLU(Device *dev, int inputChannel, int inputHeight,
                     int inputWidth)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth) {
  outputChannel_ = inputChannel;
  outputHeight_ = inputHeight;
  outputWidth_ = inputWidth;

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

LeakyReLU::LeakyReLU(Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, float negativeSlope)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth) {
  negativeSlope_ = negativeSlope;
  outputChannel_ = inputChannel;
  outputHeight_ = inputHeight;
  outputWidth_ = inputWidth;

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

LeakyReLU::~LeakyReLU() {}

void LeakyReLU::Forward(cpu::Memory *input, cpu::Memory *output) {
  float *I = reinterpret_cast<float *>(input->Ptr());
  float *O = reinterpret_cast<float *>(output->Ptr());

#pragma omp parallel for
  for (int c = 0; c < outputChannel_; ++c) {
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        int idx = c * outputHeight_ * outputWidth_ + oh * outputWidth_ + ow;
        float val = I[idx];
        if (val > 0)
          O[idx] = val;
        else
          O[idx] = val * negativeSlope_;

        // O[(c * outputHeight_ + oh) * outputWidth_ + ow] = 1;
      }
    }
  }
  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

}  // namespace op
}  // namespace cpu
