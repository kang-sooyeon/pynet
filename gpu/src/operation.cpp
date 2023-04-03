
#include <operation.h>

namespace gpu {
namespace op {

Operation::Operation(Device *dev, int inputChannel, int inputHeight,
                     int inputWidth)
    : dev_(dev),
      inputChannel_(inputChannel),
      inputHeight_(inputHeight),
      inputWidth_(inputWidth) {}

void Operation::GetOutputSize(int &outputChannel, int &outputHeight,
                              int &outputWidth) {
  outputChannel = outputChannel_;
  outputHeight = outputHeight_;
  outputWidth = outputWidth_;
}

Operation::~Operation() {}

}  // namespace op
}  // namespace gpu
