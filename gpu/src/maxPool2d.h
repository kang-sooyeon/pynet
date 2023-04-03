#ifndef MAXPOOL2d_H_
#define MAXPOOL2d_H_

#include <device.h>
#include <memory.h>
#include <operation.h>

namespace gpu {
namespace op {

class MaxPool2d : public gpu::op::Operation {
 public:
  MaxPool2d(gpu::Device *dev, int inputChannel, int inputHeight, int inputWidth,
            int kernelSize, int stride);
  virtual ~MaxPool2d();

  void Forward(gpu::Memory *input, gpu::Memory *output);

 private:
  int kernelSize_;
  int stride_;
  int inputImageSize;

  cl_kernel kernel_, kernel_img;
};

}  // namespace op
}  // namespace gpu

#endif
