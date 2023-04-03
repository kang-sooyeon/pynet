#ifndef INSTANCENORM2D_H_
#define INSTANCENORM2D_H_

#include <device.h>
#include <memory.h>
#include <operation.h>

#include <cmath>
#include <fstream>
#include <iostream>

extern cl_mem stdMem, groupSumMem, meanMem;

namespace gpu {
namespace op {

class InstanceNorm2d : public gpu::op::Operation {
 public:
  InstanceNorm2d(gpu::Device *dev, int inputChannel, int inputHeight,
                 int inputWidth, bool affine, bool reluOn);

  virtual ~InstanceNorm2d();
  void Forward(gpu::Memory *output, gpu::Memory *tmp);
  void LoadWeight(std::ifstream &fp);
  void PrintWeight();

  gpu::Memory *getWeights();
  gpu::Memory *getBias();

 private:
  int weightSize_;
  int biasSize_;
  bool reluOn_;

  gpu::Memory *bias_;
  gpu::Memory *weight_;

  cl_kernel reductionKernel_, reductionKernel_img;
  cl_kernel reductionKernel2_, reductionKernel2_img;
  cl_kernel meanKernel_, meanKernel_img;
  cl_kernel subSquareKernel_, subSquareKernel_img;
  cl_kernel stdKernel_, stdKernel_img;
  cl_kernel instanceKernel_;

  size_t gws[1];
  size_t lws[1];
};

}  // namespace op
}  // namespace gpu

#endif
