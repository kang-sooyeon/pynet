#include "transpose.h"

namespace cpu {
namespace op {

Transpose::Transpose(Device *dev, int inputChannel, int inputHeight,
                     int inputWidth, int dim0, int dim1, int dim2)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      dim0_(dim0),
      dim1_(dim1),
      dim2_(dim2) {
  outputChannel_ = inputChannel;
  outputHeight_ = inputHeight;
  outputWidth_ = inputWidth;

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

Transpose::~Transpose() {}

void Transpose::Forward(cpu::Memory *input, cpu::Memory *output) {
  float *I = reinterpret_cast<float *>(input->Ptr());
  float *O = reinterpret_cast<float *>(output->Ptr());

  for (int c = 0; c < outputChannel_; ++c) {
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        int i_idx = c * outputHeight_ * outputWidth_ + oh * outputWidth_ + ow;
        int o_idx = c * outputHeight_ * outputWidth_ + oh * outputWidth_ + ow;

        switch (dim0_) {
          case 0:
            if (dim1_ == 2)
              o_idx = c * outputHeight_ * outputWidth_ + ow * outputHeight_ +
                      oh;  // 0,2,1
            break;
          case 1:
            if (dim1_ == 2)
              o_idx = oh * outputWidth_ * outputChannel_ + ow * outputChannel_ +
                      c;  // 1,2,0
            else
              o_idx = oh * outputWidth_ * outputChannel_ + c * outputWidth_ +
                      ow;  // 1,0,2
            break;
          case 2:
            if (dim1_ == 0)
              o_idx = ow * outputChannel_ * outputHeight_ + c * outputHeight_ +
                      oh;  // 2,0,1
            else
              o_idx = ow * outputChannel_ * outputHeight_ +
                      oh * outputChannel_ + c;  // 2,1,0
            break;
        }

        O[o_idx] = I[i_idx];
      }
    }
  }
  // return output_;
}

}  // namespace op
}  // namespace cpu
