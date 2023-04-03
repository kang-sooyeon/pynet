#pragma once

#include <float.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "memory.h"
#include "operation.h"

namespace cpu {
namespace op {

class ReflectionPad2d : public cpu::op::Operation {
 public:
  ReflectionPad2d(Device *dev, int input_channel, int input_height,
                  int input_width, std::vector<int> paddings);
  ReflectionPad2d(Device *dev, int input_channel, int input_height,
                  int input_width, int paddings);
  virtual ~ReflectionPad2d();
  void Forward(cpu::Memory *input, cpu::Memory *output);

 private:
  int singlePadding_;
  std::vector<int> paddings_;
};

}  // namespace op
}  // namespace cpu
