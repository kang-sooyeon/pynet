

#ifndef CONVOLUTION2d_H_
#define CONVOLUTION2d_H_

#include <device.h>
#include <memory.h>
#include <operation.h>
#include <util.h>

#include <fstream>
#include <iostream>

extern cl_mem input_tile, filter_tile, output_tile;
extern cl_mem input_tile0, input_tile1, output_tile0, output_tile1;

namespace gpu {
namespace op {
enum class ConvolutionMode { CONV_DIRECT = 0, CONV_WINO, CONV_IMPLICIT };

class Convolution2d : public gpu::op::Operation {
 public:
  Convolution2d(gpu::Device *dev, int inputChannel, int inputHeight,
                int inputWidth, int outputChannel, int kernelSize, int stride,
                ConvolutionMode mode, bool is_split = false);
  virtual ~Convolution2d();

  void LoadWeight(std::ifstream &fp);
  void Forward(gpu::Memory *input, gpu::Memory *output);

 private:
  int kernelSize_;
  int stride_;
  int weightSize_;
  int biasSize_;

  static int cnt;

  size_t inputImageSize;
  size_t filterImageSize;

  gpu::Memory *weight_;
  gpu::Memory *bias_;

  ConvolutionMode mode_;

  cl_kernel direct_kernel, imp_kernel, imp_kernel_img, addBias_kernel;
  cl_kernel kernel_data_tile, kernel_filter_tile, kernel_gemm_opt, kernel_gemm,
      kernel_gemm_opt_img, kernel_data_untile;

  bool is_split_ = false;
  int splitN = 2;

  size_t gemm_naive_gws[3], gemm_naive_lws[3];

  size_t direct_gws[3], direct_lws[3];
  size_t imp_gws[3], imp_lws[3];
  size_t ab_gws[3], ab_lws[3];

  size_t data_gws[3], data_lws[3];
  size_t filter_gws[3], filter_lws[3];
  size_t gemm_gws[3], gemm_lws[3];
  size_t untile_gws[3], untile_lws[3];

  int H, W, C, K, P, Q, X, PQ;
};

}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {

class Convolution2d2 : public gpu::op::Operation {
 public:
  Convolution2d2(gpu::Device *dev, int inputChannel, int inputHeight,
                 int inputWidth, int outputChannel, int kernelSize, int stride,
                 ConvolutionMode mode, bool is_split = false);
  virtual ~Convolution2d2();

  void LoadWeight(std::ifstream &fp);
  void Forward(gpu::Memory *input, gpu::Memory *output);

 private:
  int kernelSize_;
  int stride_;
  int weightSize_;
  bool is_split_ = false;
  int splitN = 2;

  size_t inputImageSize;
  size_t filterImageSize;

  gpu::Memory *weight_;

  ConvolutionMode mode_;

  cl_kernel direct_kernel, imp_kernel, imp_kernel_img, imp_kernel_img2;
  cl_kernel kernel_data_tile, kernel_filter_tile, kernel_gemm_opt,
      kernel_gemm_opt_img, kernel_data_untile;

  size_t direct_gws[3], direct_lws[3];

  size_t imp_gws[3], imp_lws[3];

  size_t data_gws[3], data_lws[3];
  size_t filter_gws[3], filter_lws[3];
  size_t gemm_gws[3], gemm_lws[3];
  size_t untile_gws[3], untile_lws[3];

  int H, W, C, K, P, Q, X, PQ;
};

}  // namespace op
}  // namespace gpu

#endif
