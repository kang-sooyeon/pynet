#pragma once

#include <float.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "memory.h"
#include "operation.h"

namespace gpu {
namespace op {

class ReflectionPad2d : public gpu::op::Operation {
 public:
  ReflectionPad2d(Device *dev, int input_channel, int input_height,
                  int input_width, std::vector<int> paddings);
  ReflectionPad2d(Device *dev, int input_channel, int input_height,
                  int input_width, int paddings);
  virtual ~ReflectionPad2d();
  void Forward(gpu::Memory *input, gpu::Memory *output);
  static double timeOrg;
  static double timeTest1;
  static double timeTest2;

  static double kernelLaunchTime;
  static int launchCnt;

  static void printLaunchStatus();

 private:
  int inputImageSize;
  int singlePadding_;
  std::vector<int> paddings_;
  cl_kernel kernel_, kernel_img;
};

}  // namespace op
}  // namespace gpu
