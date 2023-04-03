#include "mseLoss.h"

namespace cpu {
namespace op {

MSELoss::MSELoss(Device *dev, int inputChannel, int inputHeight, int inputWidth,
                 int reduction)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      reduction_(reduction) {
  // reduction
  // 0 = mean
  // 1 = sum
  outputChannel_ = 1;
  outputHeight_ = 1;
  outputWidth_ = 1;

  size_ = inputChannel * inputHeight * inputWidth;

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

MSELoss::~MSELoss() {}

void MSELoss::Forward(cpu::Memory *input1, cpu::Memory *input2,
                      cpu::Memory *output) {
  float *I1 = reinterpret_cast<float *>(input1->Ptr());
  float *I2 = reinterpret_cast<float *>(input2->Ptr());
  float *O = reinterpret_cast<float *>(output->Ptr());
  register float sum = 0;
  register float tmp = 0;

  for (int c = 0; c < inputChannel_; ++c) {
    for (int oh = 0; oh < inputHeight_; ++oh) {
      for (int ow = 0; ow < inputWidth_; ++ow) {
        tmp = I1[(c * inputHeight_ + oh) * inputWidth_ + ow] -
              I2[(c * inputHeight_ + oh) * inputWidth_ + ow];
        tmp = tmp * tmp;
        // std::cout << "target : " << I1[(c * inputHeight_ + oh) * inputWidth_
        // + ow] << " , " << I2[(c * inputHeight_ + oh) * inputWidth_ + ow] <<
        // std::endl; std::cout << "tmp : " << tmp << std::endl;
        sum += tmp;
      }
    }
  }
  // std::cout << std::endl
  // 		  << "final : " << sum << std::endl;
  // mean reduction
  if (reduction_ == 0) sum = sum / size_;

  O[0] = sum;

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

}  // namespace op
}  // namespace cpu
