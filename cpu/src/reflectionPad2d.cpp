#include "reflectionPad2d.h"

#include <omp.h>

namespace cpu {
namespace op {

ReflectionPad2d::ReflectionPad2d(Device *dev, int inputChannel, int inputHeight,
                                 int inputWidth, std::vector<int> paddings)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      paddings_(paddings) {
  // padding range check
  if (paddings[0] >= inputWidth_ || paddings[1] >= inputWidth_ ||
      paddings[2] >= inputHeight_ || paddings[3] >= inputHeight_) {
    std::cerr
        << "Padding size should be less than the corresponding input dimension"
        << std::endl;
    abort();
  }

  outputChannel_ = inputChannel_;
  outputHeight_ = inputHeight_ + paddings_[2] +
                  paddings_[3];  // height = heitht + topPad + bottomPad
  outputWidth_ = inputWidth_ + paddings_[0] +
                 paddings_[1];  // width = width + leftPad + rightPad

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

ReflectionPad2d::ReflectionPad2d(Device *dev, int inputChannel, int inputHeight,
                                 int inputWidth, int paddings)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      singlePadding_(paddings) {
  paddings_ = {singlePadding_, singlePadding_, singlePadding_, singlePadding_};
  // padding range check
  if (paddings_[0] >= inputWidth_ || paddings_[1] >= inputWidth_ ||
      paddings_[2] >= inputHeight_ || paddings_[3] >= inputHeight_) {
    std::cerr
        << "Padding size should be less than the corresponding input dimension"
        << std::endl;
    abort();
  }

  outputChannel_ = inputChannel_;
  outputHeight_ = inputHeight_ + paddings_[2] +
                  paddings_[3];  // height = heitht + topPad + bottomPad
  outputWidth_ = inputWidth_ + paddings_[0] +
                 paddings_[1];  // width = width + leftPad + rightPad

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

ReflectionPad2d::~ReflectionPad2d() {
  // delete output_;
}

void ReflectionPad2d::Forward(cpu::Memory *input, cpu::Memory *output) {
  float *I = reinterpret_cast<float *>(input->Ptr());
  float *O = reinterpret_cast<float *>(output->Ptr());

  /*
  for (int k = 0; k < outputChannel_; ++k) {
          for (int oh = 0; oh < outputHeight_; ++oh) {
                  for (int ow = 0; ow < outputWidth_; ++ow) {
                          O[(k * outputHeight_ + oh) * outputWidth_ + ow] = 1;
                  }
          }
  }
  return output_;
  */

  // area position
  // 0 |  2  | 1
  // 6 | pad | 7
  // 3 |  5  | 4

#pragma omp parallel for
  for (int c = 0; c < outputChannel_; ++c) {
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        // pad added area
        if (oh < paddings_[2] || ow < paddings_[0] ||
            oh > (paddings_[2] + (inputHeight_ - 1)) ||
            ow > (paddings_[0] + (inputWidth_ - 1))) {
          O[(c * outputHeight_ + oh) * outputWidth_ + ow] = -1;
          if (oh < paddings_[2]) {
            if (ow < paddings_[0])  // 0
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ + (paddings_[2] - oh)) * inputWidth_ +
                    (paddings_[0] - ow)];
            } else if (ow > (paddings_[0] + (inputWidth_ - 1)))  // 1
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ + (paddings_[2] - oh)) * inputWidth_ +
                    (2 * inputWidth_ + paddings_[0] - ow - 2)];
            } else  // 2
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ + (paddings_[2] - oh)) * inputWidth_ +
                    (ow - paddings_[0])];
            }
          } else if (oh > (paddings_[2] + (inputHeight_ - 1))) {
            if (ow < paddings_[0])  // 3
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ +
                     (2 * inputHeight_ + paddings_[2] - oh - 2)) *
                        inputWidth_ +
                    (paddings_[0] - ow)];
            } else if (ow > (paddings_[0] + (inputWidth_ - 1)))  // 4
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ +
                     (2 * inputHeight_ + paddings_[2] - oh - 2)) *
                        inputWidth_ +
                    (2 * inputWidth_ + paddings_[0] - ow - 2)];
            } else  // 5
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ +
                     (2 * inputHeight_ + paddings_[2] - oh - 2)) *
                        inputWidth_ +
                    (ow - paddings_[0])];
            }
          } else {
            if (ow < paddings_[0])  // 6
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ + (oh - paddings_[2])) * inputWidth_ +
                    (paddings_[0] - ow)];
            } else  // 7
            {
              O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                  I[(c * inputHeight_ + (oh - paddings_[2])) * inputWidth_ +
                    (2 * inputWidth_ + paddings_[0] - ow - 2)];
            }
          }
        }
        // pad origin area
        else {
          O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
              I[(c * inputHeight_ + (oh - paddings_[2])) * inputWidth_ +
                (ow - paddings_[0])];
        }
      }
    }
  }
  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

}  // namespace op
}  // namespace cpu
