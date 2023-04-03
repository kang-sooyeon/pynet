
#include "operation.h"

namespace cpu {
namespace op {

Operation::Operation(cpu::Device *dev, int inputChannel, int inputHeight,
                     int inputWidth)
    : dev_(dev),
      inputChannel_(inputChannel),
      inputHeight_(inputHeight),
      inputWidth_(inputWidth) {}

/*
int Operation::GetOutputSize()
{
        return outputChannel_ * outputHeight_ * outputWidth_;
}
*/

int Operation::GetOutputSizeAll() {
  return outputChannel_ * outputHeight_ * outputWidth_ * sizeof(float);
}

void Operation::GetOutputSize(int &outputChannel, int &outputHeight,
                              int &outputWidth) {
  outputChannel = outputChannel_;
  outputHeight = outputHeight_;
  outputWidth = outputWidth_;
}

/*
void Operation::PrintOutput(cpu::Memory* output)
{
        float *O = reinterpret_cast<float *>(output->Ptr());

        for (int c = 0; c < outputChannel_; ++c)
        {
                for (int oh = 0; oh < outputHeight_; ++oh)
                {
                        for (int ow = 0; ow < outputWidth_; ++ow)
                        {
                                int idx = (c * outputHeight_ + oh) *
outputWidth_ + ow; std::cout << O[idx] << ", ";
                        }
                        std::cout << std::endl;
                }
                std::cout << std::endl;
        }

        for( int i = 0; i < 10; i++ )
                std::cout << O[i] << ", ";
}
*/

/*
cpu::Memory *Operation::GetOutput()
{
        return output_;
}
*/

Operation::~Operation() {}

}  // namespace op
}  // namespace cpu
