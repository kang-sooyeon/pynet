#include "cat.h"

#include <omp.h>

namespace cpu {
namespace op {

// Cat
Cat::Cat(Device *dev, int inputChannel1, int inputHeight1, int inputWidth1,
         int inputChannel2, int inputHeight2, int inputWidth2, int dim)
    : cpu::op::Operation(dev, inputChannel1, inputHeight1, inputWidth1),
      inputChannel2_(inputChannel2),
      inputHeight2_(inputHeight2),
      inputWidth2_(inputWidth2),
      dim_(dim) {
  if (dim == 0) {
    // do notiong. we don't care about that case
  } else if (dim == 1) {
    if (inputWidth_ != inputWidth2_ || inputHeight_ != inputHeight2_) {
      std::cerr << "Invalid Input Tensor Size <Cat-channel>" << std::endl;
      std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " , "
                << inputWidth_ << std::endl;
      std::cerr << "<" << inputChannel2_ << " , " << inputHeight2_ << " , "
                << inputWidth2_ << std::endl;
      abort();
    }

    outputChannel_ = inputChannel_ + inputChannel2_;
    outputHeight_ = inputHeight_;
    outputWidth_ = inputWidth_;
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
  } else {
    std::cerr << "Invalid Dimension Parameter!" << std::endl;
    abort();
  }

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

Cat::~Cat() {
  // delete output_;
}

void Cat::Forward(cpu::Memory *input1, cpu::Memory *input2,
                  cpu::Memory *output) {
  float *I1 = reinterpret_cast<float *>(input1->Ptr());
  float *I2 = reinterpret_cast<float *>(input2->Ptr());
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

  //#pragma omp parallel for
  for (int c = 0; c < outputChannel_; ++c) {
    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow) {
        if (dim_ == 1) {
          if (c < inputChannel_) {
            O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                I1[(c * inputHeight_ + oh) * inputWidth_ + ow];
          } else {
            O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                I2[((c - inputChannel_) * inputHeight2_ + oh) * inputWidth2_ +
                   ow];
          }
        } else if (dim_ == 2) {
          if (oh < inputHeight_) {
            O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                I1[(c * inputHeight_ + oh) * inputWidth_ + ow];
          } else {
            O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                I2[(c * inputHeight2_ + (oh - inputHeight_)) * inputWidth2_ +
                   ow];
          }
        } else {
          if (ow < inputWidth_) {
            O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                I1[(c * inputHeight_ + oh) * inputWidth_ + ow];
          } else {
            O[(c * outputHeight_ + oh) * outputWidth_ + ow] =
                I2[(c * inputHeight2_ + oh) * inputWidth2_ +
                   (ow - inputWidth_)];
          }
        }
      }
    }
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

/*
// Cat3
Cat3::Cat3(Device *dev,
                 int inputChannel1, int inputHeight1, int inputWidth1,
                 int inputChannel2, int inputHeight2, int inputWidth2)
                 int inputChannel3, int inputHeight3, int inputWidth3, int dim)
        : cpu::op::Operation(dev, inputChannel1, inputHeight1, inputWidth1),
inputChannel2_(inputChannel2), inputHeight2_(inputHeight2),
inputWidth2_(inputWidth2), inputChannel3_(inputChannel3),
inputHeight3_(inputHeight3), inputWidth3_(inputWidth3), dim_(dim)
{

        if (dim == 0)
        {
                // do notiong. we don't care about that case
        }
        else if (dim == 1)
        {

                if (inputWidth_ != inputWidth2_ || inputHeight_ !=
inputHeight2_)
                {
                        std::cerr << "Invalid Input Tensor Size <Cat-channel>"
<< std::endl; std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " ,
" << inputWidth_ << std::endl; std::cerr << "<" << inputChannel2_ << " , " <<
inputHeight2_ << " , " << inputWidth2_ << std::endl; abort();
                }

                outputChannel_ = inputChannel_ + inputChannel2_;
                outputHeight_ = inputHeight_;
                outputWidth_ = inputWidth_;
        }
        else if (dim == 2)
        {
                if (inputChannel_ != inputChannel2_ || inputWidth_ !=
inputWidth2_)
                {
                        std::cerr << "Invalid Input Tensor Size <Cat-height>" <<
std::endl; std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " , "
<< inputWidth_ << std::endl; std::cerr << "<" << inputChannel2_ << " , " <<
inputHeight2_ << " , " << inputWidth2_ << std::endl; abort();
                }

                outputChannel_ = inputChannel_;
                outputHeight_ = inputHeight_ + inputHeight2_;
                outputWidth_ = inputWidth_;
        }
        else if (dim == 3)
        {
                if (inputChannel_ != inputChannel2_ || inputHeight_ !=
inputHeight2_)
                {
                        std::cerr << "Invalid Input Tensor Size <Cat-width>" <<
std::endl; std::cerr << "<" << inputChannel_ << " , " << inputHeight_ << " , "
<< inputWidth_ << std::endl; std::cerr << "<" << inputChannel2_ << " , " <<
inputHeight2_ << " , " << inputWidth2_ << std::endl; abort();
                }

                outputChannel_ = inputChannel_;
                outputHeight_ = inputHeight_;
                outputWidth_ = inputWidth_ + inputWidth2_;
        }
        else
        {
                std::cerr << "Invalid Dimension Parameter!" << std::endl;
                abort();
        }

        output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
outputWidth_ * sizeof(float));
}

Cat::~Cat()
{
        delete output_;
}

cpu::Memory *Cat::Forward(cpu::Memory *input1, cpu::Memory *input2)
{
        float *I1 = reinterpret_cast<float *>(input1->Ptr());
        float *I2 = reinterpret_cast<float *>(input2->Ptr());
        float *O = reinterpret_cast<float *>(output_->Ptr());

        for (int c = 0; c < outputChannel_; ++c)
        {
                for (int oh = 0; oh < outputHeight_; ++oh)
                {
                        for (int ow = 0; ow < outputWidth_; ++ow)
                        {
                                if (dim_ == 1)
                                {
                                        if (c < inputChannel_)
                                        {
                                                O[(c * outputHeight_ + oh) *
outputWidth_ + ow] = I1[(c * inputHeight_ + oh) * inputWidth_ + ow];
                                        }
                                        else
                                        {
                                                O[(c * outputHeight_ + oh) *
outputWidth_ + ow] = I2[((c - inputChannel_) * inputHeight2_ + oh) *
inputWidth2_ + ow];
                                        }
                                }
                                else if (dim_ == 2)
                                {
                                        if (oh < inputHeight_)
                                        {
                                                O[(c * outputHeight_ + oh) *
outputWidth_ + ow] = I1[(c * inputHeight_ + oh) * inputWidth_ + ow];
                                        }
                                        else
                                        {
                                                O[(c * outputHeight_ + oh) *
outputWidth_ + ow] = I2[(c * inputHeight2_ + (oh - inputHeight_)) * inputWidth2_
+ ow];
                                        }
                                }
                                else
                                {
                                        if (ow < inputWidth_)
                                        {
                                                O[(c * outputHeight_ + oh) *
outputWidth_ + ow] = I1[(c * inputHeight_ + oh) * inputWidth_ + ow];
                                        }
                                        else
                                        {
                                                O[(c * outputHeight_ + oh) *
outputWidth_ + ow] = I2[(c * inputHeight2_ + oh) * inputWidth2_ + (ow -
inputWidth_)];
                                        }
                                }
                        }
                }
        }

        return output_;
}

*/

}  // namespace op
}  // namespace cpu
