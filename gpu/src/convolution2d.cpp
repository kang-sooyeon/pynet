
#include "convolution2d.h"

#include <stdlib.h>
#include <time.h>

#include <cmath>
#include <fstream>

using namespace std;
using half_float::half;
using namespace half_float::literal;

namespace gpu {
namespace op {

int Convolution2d::cnt = 0;

Convolution2d::Convolution2d(gpu::Device *dev, int inputChannel,
                             int inputHeight, int inputWidth, int outputChannel,
                             int kernelSize, int stride, ConvolutionMode mode,
                             bool is_split)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride),
      mode_(mode),
      is_split_(is_split) {
  outputChannel_ = outputChannel;
  outputHeight_ = (inputHeight_ - kernelSize_) / stride_ + 1;
  outputWidth_ = (inputWidth_ - kernelSize_) / stride_ + 1;
  weightSize_ = outputChannel_ * inputChannel_ * kernelSize * kernelSize *
                sizeof(DATA_TYPE);
  weight_ = new gpu::Memory(dev, weightSize_);
  biasSize_ = outputChannel_ * sizeof(DATA_TYPE);
  bias_ = new gpu::Memory(dev, biasSize_);

  cl_int err = 0;

  if (mode_ == ConvolutionMode::CONV_DIRECT) {
    direct_kernel =
        clCreateKernel(dev_->ClDnnHandle()->program, "convolution2d2", &err);
    CheckCl(err);

    direct_gws[0] = outputWidth_;
    direct_gws[1] = outputHeight_;
    direct_gws[2] = outputChannel_;
    direct_lws[0] = 16;
    direct_lws[1] = 4;
    direct_lws[2] = 4;

    direct_gws[0] =
        (direct_gws[0] + direct_lws[0] - 1) / direct_lws[0] * direct_lws[0];
    direct_gws[1] =
        (direct_gws[1] + direct_lws[1] - 1) / direct_lws[1] * direct_lws[1];
    direct_gws[2] =
        (direct_gws[2] + direct_lws[2] - 1) / direct_lws[2] * direct_lws[2];
  } else if (mode_ == ConvolutionMode::CONV_WINO) {
    if (kernelSize_ == 3) {
      kernel_data_tile =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino3_data_tile", &err);
      CheckCl(err);
      kernel_filter_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino3_filter_tile", &err);
      CheckCl(err);
      kernel_gemm_opt =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino3_gemm_opt", &err);
      CheckCl(err);
      kernel_gemm =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino3_gemm", &err);
      CheckCl(err);
      kernel_data_untile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino3_data_untile", &err);
      CheckCl(err);

      inputImageSize =
          inputChannel_ * (inputHeight_ / 4) * (inputWidth_ / 4) * 36;
      filterImageSize = inputChannel_ * outputChannel_ * 36;

      H = inputHeight_;
      W = inputWidth_;
      C = inputChannel_;
      K = outputChannel_;
      P = (H + 1) / 4;
      Q = (W + 1) / 4;
      X = 36;
      PQ = P * Q;

      data_lws[0] = 16;
      data_lws[1] = 16;
      data_lws[2] = 1;
      data_gws[0] = ((Q + 15) / 16) * 16;
      data_gws[1] = ((P + 15) / 16) * 16;
      data_gws[2] = C;

      int filter_BSZ = 128;
      filter_lws[0] = filter_BSZ;
      filter_lws[1] = filter_lws[2] = 1;
      filter_gws[0] = (((C * K) + filter_BSZ) / filter_BSZ) * filter_BSZ;
      filter_gws[1] = filter_gws[2] = 1;

      int gemm_TILE_M = 32;
      int gemm_TILE_N = 32;
      int gemm_BSZ = 64;
      gemm_gws[0] = X * ((P * Q + gemm_TILE_N - 1) / gemm_TILE_N) *
                    ((K + gemm_TILE_M - 1) / gemm_TILE_M) * gemm_BSZ;
      gemm_gws[1] = gemm_gws[2] = 1;
      gemm_lws[0] = gemm_BSZ;
      gemm_lws[1] = gemm_lws[2] = 1;

      int untile_BSZ = 128;
      untile_gws[0] = ((K * P * Q + untile_BSZ - 1) / untile_BSZ) * untile_BSZ;
      untile_gws[1] = untile_gws[2] = 1;
      untile_lws[0] = untile_BSZ;
      untile_lws[1] = untile_lws[2] = 1;

    } else {
      if (!is_split_) {
        inputImageSize =
            inputChannel_ * (inputHeight_ / 4) * (inputWidth_ / 4) * 64;
        filterImageSize = inputChannel_ * outputChannel_ * 64;

        kernel_data_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino5_data_tile", &err);
        CheckCl(err);
        kernel_filter_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                            "conv_wino5_filter_tile", &err);
        CheckCl(err);
        kernel_gemm_opt = clCreateKernel(dev_->ClDnnHandle()->program,
                                         "conv_wino5_gemm_opt", &err);
        CheckCl(err);
        kernel_gemm_opt_img = clCreateKernel(dev_->ClDnnHandle()->program,
                                             "conv_wino5_gemm_opt_img", &err);
        CheckCl(err);
        kernel_data_untile = clCreateKernel(dev_->ClDnnHandle()->program,
                                            "conv_wino5_data_untile", &err);
        CheckCl(err);
      } else {
        inputImageSize =
            inputChannel_ * (inputHeight_ / 8) * (inputWidth_ / 4) * 64;
        filterImageSize = inputChannel_ * outputChannel_ * 64;

        kernel_data_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino5_data_tile_split", &err);
        CheckCl(err);
        kernel_filter_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                            "conv_wino5_filter_tile", &err);
        CheckCl(err);
        kernel_gemm_opt = clCreateKernel(dev_->ClDnnHandle()->program,
                                         "conv_wino5_gemm_opt_split", &err);
        CheckCl(err);
        kernel_gemm_opt_img = clCreateKernel(dev_->ClDnnHandle()->program,
                                             "conv_wino5_gemm_opt_split_img", &err);
        CheckCl(err);
        kernel_data_untile = clCreateKernel(dev_->ClDnnHandle()->program,
                                            "conv_wino5_data_untile_split", &err);
        CheckCl(err);
      }

      H = inputHeight_;
      W = inputWidth_;
      C = inputChannel_;
      K = outputChannel_;
      P = (int)ceil((H - 4) / 4.0f);
      if (is_split_) P /= 2;
      Q = (int)ceil((W - 4) / 4.0f);
      X = 64;
      PQ = P * Q;

      // best
      data_lws[0] = 16;
      data_lws[1] = 16;
      data_lws[2] = 1;
      data_gws[0] = ((Q + 15) / 16) * 16;
      data_gws[1] = ((P + 15) / 16) * 16;
      data_gws[2] = C;

      int filter_BSZ = 128;
      filter_lws[0] = filter_BSZ;
      filter_lws[1] = filter_lws[2] = 1;
      filter_gws[0] = (((C * K) + filter_BSZ) / filter_BSZ) * filter_BSZ;
      filter_gws[1] = filter_gws[2] = 1;

      int gemm_TILE_M = 32;
      int gemm_TILE_N = 32;
      int gemm_BSZ = 64;
      gemm_gws[0] = X * ((P * Q + gemm_TILE_N - 1) / gemm_TILE_N) *
                    ((K + gemm_TILE_M - 1) / gemm_TILE_M) * gemm_BSZ;
      gemm_gws[1] = gemm_gws[2] = 1;
      gemm_lws[0] = gemm_BSZ;
      gemm_lws[1] = gemm_lws[2] = 1;

      int untile_BSZ = 128;
      untile_gws[0] = ((K * P * Q + untile_BSZ - 1) / untile_BSZ) * untile_BSZ;
      untile_gws[1] = untile_gws[2] = 1;
      untile_lws[0] = untile_BSZ;
      untile_lws[1] = untile_lws[2] = 1;
    }
  } else if (mode_ == ConvolutionMode::CONV_IMPLICIT) {
    imp_kernel = clCreateKernel(dev_->ClDnnHandle()->program,
                                "conv_implicit_gemm", &err);
    CheckCl(err);
    imp_kernel_img = clCreateKernel(dev_->ClDnnHandle()->program,
                                     "conv_implicit_gemm_img", &err);
    CheckCl(err);
    addBias_kernel =
        clCreateKernel(dev_->ClDnnHandle()->program, "addBias", &err);
    CheckCl(err);

    inputImageSize = inputChannel_ * inputHeight_ * inputWidth_;
    filterImageSize =
        inputChannel_ * outputChannel_ * kernelSize_ * kernelSize_;

    int imp_TILE_M = 32;
    int imp_TILE_N = 32;
    int imp_BSZ = 64;
    imp_gws[1] = 1;
    imp_gws[2] = 1;
    imp_lws[1] = 1;
    imp_lws[2] = 1;
    imp_lws[0] = imp_BSZ;
    imp_gws[0] =
        ((outputHeight_ * outputWidth_ + imp_TILE_N - 1) / imp_TILE_N) *
        ((outputChannel_ + imp_TILE_M - 1) / imp_TILE_M) * imp_lws[0];

    ab_gws[0] = outputWidth_;
    ab_gws[1] = outputHeight_;
    ab_gws[2] = outputChannel_;
    ab_lws[0] = 64;
    ab_lws[1] = 4;
    ab_lws[2] = 1;
    ab_gws[0] = (ab_gws[0] + ab_lws[0] - 1) / ab_lws[0] * ab_lws[0];
    ab_gws[1] = (ab_gws[1] + ab_lws[1] - 1) / ab_lws[1] * ab_lws[1];
    ab_gws[2] = (ab_gws[2] + ab_lws[2] - 1) / ab_lws[2] * ab_lws[2];
  } else {
    cout << "no algorithm" << endl;
    exit(-1);
  }
}

void Convolution2d::LoadWeight(std::ifstream &fp) {
  char *weight_H = (char *)malloc(weightSize_);
  fp.read(weight_H, weightSize_);
  weight_->Set(weight_H);
  delete weight_H;

  char *bias_H = (char *)malloc(biasSize_);
  fp.read(bias_H, biasSize_);
  bias_->Set(bias_H);
  delete bias_H;
}

void Convolution2d::Forward(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();
  cl_mem weightMem = weight_->Ptr();
  cl_mem biasMem = bias_->Ptr();

  cl_mem inputImage, filterImage;
  double measured, val;

  cl_int err = 0;
  if (mode_ == ConvolutionMode::CONV_DIRECT) {
    cl_event event;

    CheckCl(clSetKernelArg(direct_kernel, 0, sizeof(cl_mem), &inputMem));
    CheckCl(clSetKernelArg(direct_kernel, 1, sizeof(cl_mem), &outputMem));
    CheckCl(clSetKernelArg(direct_kernel, 2, sizeof(cl_mem), &weightMem));
    CheckCl(clSetKernelArg(direct_kernel, 3, sizeof(cl_mem), &biasMem));
    CheckCl(clSetKernelArg(direct_kernel, 4, sizeof(cl_int), &inputChannel_));
    CheckCl(clSetKernelArg(direct_kernel, 5, sizeof(cl_int), &inputHeight_));
    CheckCl(clSetKernelArg(direct_kernel, 6, sizeof(cl_int), &inputWidth_));
    CheckCl(clSetKernelArg(direct_kernel, 7, sizeof(cl_int), &outputChannel_));
    CheckCl(clSetKernelArg(direct_kernel, 8, sizeof(cl_int), &outputHeight_));
    CheckCl(clSetKernelArg(direct_kernel, 9, sizeof(cl_int), &outputWidth_));
    CheckCl(clSetKernelArg(direct_kernel, 10, sizeof(cl_int), &kernelSize_));

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, direct_kernel, 3,
                                   NULL, direct_gws, direct_lws, 0, NULL,
                                   &event));
  } else if (mode_ == ConvolutionMode::CONV_IMPLICIT) {
    if (inputImageSize > 134217728 * 2) {
      CheckCl(clSetKernelArg(imp_kernel, 0, sizeof(cl_mem), &inputMem));
      CheckCl(clSetKernelArg(imp_kernel, 1, sizeof(cl_mem), &outputMem));
      CheckCl(clSetKernelArg(imp_kernel, 2, sizeof(cl_mem), &weightMem));
      CheckCl(clSetKernelArg(imp_kernel, 3, sizeof(cl_int), &inputChannel_));
      CheckCl(clSetKernelArg(imp_kernel, 4, sizeof(cl_int), &outputChannel_));
      CheckCl(clSetKernelArg(imp_kernel, 5, sizeof(cl_int), &inputHeight_));
      CheckCl(clSetKernelArg(imp_kernel, 6, sizeof(cl_int), &inputWidth_));
      CheckCl(clSetKernelArg(imp_kernel, 7, sizeof(cl_int), &outputHeight_));
      CheckCl(clSetKernelArg(imp_kernel, 8, sizeof(cl_int), &outputWidth_));
      CheckCl(clSetKernelArg(imp_kernel, 9, sizeof(cl_int), &kernelSize_));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, imp_kernel, 3,
                                     NULL, imp_gws, imp_lws, 0, NULL, NULL));

    } else {
      inputImage = input->createImage2(inputImageSize);
      filterImage = weight_->createImage2(filterImageSize);

      CheckCl(clSetKernelArg(imp_kernel_img, 0, sizeof(cl_mem), &inputImage));
      CheckCl(clSetKernelArg(imp_kernel_img, 1, sizeof(cl_mem), &outputMem));
      CheckCl(clSetKernelArg(imp_kernel_img, 2, sizeof(cl_mem), &filterImage));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 3, sizeof(cl_int), &inputChannel_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 4, sizeof(cl_int), &outputChannel_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 5, sizeof(cl_int), &inputHeight_));
      CheckCl(clSetKernelArg(imp_kernel_img, 6, sizeof(cl_int), &inputWidth_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 7, sizeof(cl_int), &outputHeight_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 8, sizeof(cl_int), &outputWidth_));
      CheckCl(clSetKernelArg(imp_kernel_img, 9, sizeof(cl_int), &kernelSize_));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     imp_kernel_img, 3, NULL, imp_gws, imp_lws,
                                     0, NULL, NULL));
    }

    CheckCl(clSetKernelArg(addBias_kernel, 0, sizeof(cl_mem), &outputMem));
    CheckCl(clSetKernelArg(addBias_kernel, 1, sizeof(cl_mem), &biasMem));
    CheckCl(clSetKernelArg(addBias_kernel, 2, sizeof(cl_int), &outputChannel_));
    CheckCl(clSetKernelArg(addBias_kernel, 3, sizeof(cl_int), &outputHeight_));
    CheckCl(clSetKernelArg(addBias_kernel, 4, sizeof(cl_int), &outputWidth_));

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, addBias_kernel,
                                   3, NULL, ab_gws, ab_lws, 0, NULL, NULL));

  } else if (mode_ == ConvolutionMode::CONV_WINO) {
    if (!is_split_) {
      //////// data tile
      CheckCl(clSetKernelArg(kernel_data_tile, 0, sizeof(cl_mem), &inputMem));
      CheckCl(clSetKernelArg(kernel_data_tile, 1, sizeof(cl_mem), &input_tile));
      CheckCl(clSetKernelArg(kernel_data_tile, 2, sizeof(cl_int), &C));
      CheckCl(clSetKernelArg(kernel_data_tile, 3, sizeof(cl_int), &H));
      CheckCl(clSetKernelArg(kernel_data_tile, 4, sizeof(cl_int), &W));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     kernel_data_tile, 3, NULL, data_gws,
                                     data_lws, 0, NULL, NULL));

      //////// filter tile
      CheckCl(
          clSetKernelArg(kernel_filter_tile, 0, sizeof(cl_mem), &weightMem));
      CheckCl(
          clSetKernelArg(kernel_filter_tile, 1, sizeof(cl_mem), &filter_tile));
      CheckCl(clSetKernelArg(kernel_filter_tile, 2, sizeof(cl_int), &C));
      CheckCl(clSetKernelArg(kernel_filter_tile, 3, sizeof(cl_int), &K));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     kernel_filter_tile, 3, NULL, filter_gws,
                                     filter_lws, 0, NULL, NULL));

      
      // gemm opt
      if(PQ % 2 == 0){
        CheckCl(clSetKernelArg(kernel_gemm_opt, 0, sizeof(cl_mem), &filter_tile));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 1, sizeof(cl_mem), &input_tile));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 2, sizeof(cl_mem), &output_tile));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 3, sizeof(cl_int), &X));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 4, sizeof(cl_int), &K));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 5, sizeof(cl_int), &PQ));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 6, sizeof(cl_int), &C));

        CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                      kernel_gemm_opt, 3, NULL, gemm_gws,
                                      gemm_lws, 0, NULL, NULL));
      }
      else{
        CheckCl(clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &filter_tile));
        CheckCl(clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &input_tile));
        CheckCl(clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &output_tile));
        CheckCl(clSetKernelArg(kernel_gemm, 3, sizeof(cl_int), &X));
        CheckCl(clSetKernelArg(kernel_gemm, 4, sizeof(cl_int), &K));
        CheckCl(clSetKernelArg(kernel_gemm, 5, sizeof(cl_int), &PQ));
        CheckCl(clSetKernelArg(kernel_gemm, 6, sizeof(cl_int), &C));

        CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                      kernel_gemm, 3, NULL, gemm_gws,
                                      gemm_lws, 0, NULL, NULL));
      }      

      /////// untile
      CheckCl(
          clSetKernelArg(kernel_data_untile, 0, sizeof(cl_mem), &output_tile));
      CheckCl(
          clSetKernelArg(kernel_data_untile, 1, sizeof(cl_mem), &outputMem));
      CheckCl(clSetKernelArg(kernel_data_untile, 2, sizeof(cl_mem), &biasMem));
      CheckCl(clSetKernelArg(kernel_data_untile, 3, sizeof(cl_int), &K));
      CheckCl(clSetKernelArg(kernel_data_untile, 4, sizeof(cl_int), &H));
      CheckCl(clSetKernelArg(kernel_data_untile, 5, sizeof(cl_int), &W));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     kernel_data_untile, 3, NULL, untile_gws,
                                     untile_lws, 0, NULL, NULL));

    } else {  // is_split

      // filter tile
      CheckCl(
          clSetKernelArg(kernel_filter_tile, 0, sizeof(cl_mem), &weightMem));
      CheckCl(
          clSetKernelArg(kernel_filter_tile, 1, sizeof(cl_mem), &filter_tile));
      CheckCl(clSetKernelArg(kernel_filter_tile, 2, sizeof(cl_int), &C));
      CheckCl(clSetKernelArg(kernel_filter_tile, 3, sizeof(cl_int), &K));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     kernel_filter_tile, 3, NULL, filter_gws,
                                     filter_lws, 0, NULL, NULL));

      for (int split_id = 0; split_id < splitN; split_id++) {
        // data tile
        CheckCl(clSetKernelArg(kernel_data_tile, 0, sizeof(cl_mem), &inputMem));
        CheckCl(
            clSetKernelArg(kernel_data_tile, 1, sizeof(cl_mem), &input_tile));
        CheckCl(clSetKernelArg(kernel_data_tile, 2, sizeof(cl_int), &C));
        CheckCl(clSetKernelArg(kernel_data_tile, 3, sizeof(cl_int), &H));
        CheckCl(clSetKernelArg(kernel_data_tile, 4, sizeof(cl_int), &W));
        CheckCl(clSetKernelArg(kernel_data_tile, 5, sizeof(cl_int), &split_id));

        CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                       kernel_data_tile, 3, NULL, data_gws,
                                       data_lws, 0, NULL, NULL));

        // gemm opt
        CheckCl(
            clSetKernelArg(kernel_gemm_opt, 0, sizeof(cl_mem), &filter_tile));
        CheckCl(
            clSetKernelArg(kernel_gemm_opt, 1, sizeof(cl_mem), &input_tile));
        CheckCl(
            clSetKernelArg(kernel_gemm_opt, 2, sizeof(cl_mem), &output_tile));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 3, sizeof(cl_int), &X));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 4, sizeof(cl_int), &K));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 5, sizeof(cl_int), &PQ));
        CheckCl(clSetKernelArg(kernel_gemm_opt, 6, sizeof(cl_int), &C));

        CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                       kernel_gemm_opt, 3, NULL, gemm_gws,
                                       gemm_lws, 0, NULL, NULL));

        // untile
        CheckCl(clSetKernelArg(kernel_data_untile, 0, sizeof(cl_mem),
                               &output_tile));
        CheckCl(
            clSetKernelArg(kernel_data_untile, 1, sizeof(cl_mem), &outputMem));
        CheckCl(
            clSetKernelArg(kernel_data_untile, 2, sizeof(cl_mem), &biasMem));
        CheckCl(clSetKernelArg(kernel_data_untile, 3, sizeof(cl_int), &K));
        CheckCl(clSetKernelArg(kernel_data_untile, 4, sizeof(cl_int), &H));
        CheckCl(clSetKernelArg(kernel_data_untile, 5, sizeof(cl_int), &W));
        CheckCl(
            clSetKernelArg(kernel_data_untile, 6, sizeof(cl_int), &split_id));

        CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                       kernel_data_untile, 3, NULL, untile_gws,
                                       untile_lws, 0, NULL, NULL));
      }

    }  // end is_split
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

Convolution2d::~Convolution2d() {
  delete weight_;
  delete bias_;
}

}  // namespace op
}  // namespace gpu

namespace gpu {
namespace op {
Convolution2d2::Convolution2d2(gpu::Device *dev, int inputChannel,
                               int inputHeight, int inputWidth,
                               int outputChannel, int kernelSize, int stride,
                               ConvolutionMode mode, bool is_split)
    : gpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      kernelSize_(kernelSize),
      stride_(stride),
      mode_(mode),
      is_split_(is_split) {
  outputChannel_ = outputChannel;
  outputHeight_ = (inputHeight_ - kernelSize_) / stride_ + 1;
  outputWidth_ = (inputWidth_ - kernelSize_) / stride_ + 1;
  weightSize_ = outputChannel_ * inputChannel_ * kernelSize * kernelSize *
                sizeof(DATA_TYPE);
  weight_ = new gpu::Memory(dev, weightSize_);

  cl_int err = 0;

  if (mode_ == ConvolutionMode::CONV_DIRECT) {
    direct_kernel =
        clCreateKernel(dev_->ClDnnHandle()->program, "convolution2d", &err);
    CheckCl(err);

    direct_gws[0] = outputWidth_;
    direct_gws[1] = outputHeight_;
    direct_gws[2] = outputChannel_;
    direct_lws[0] = 16;
    direct_lws[1] = 4;
    direct_lws[2] = 4;

    direct_gws[0] =
        (direct_gws[0] + direct_lws[0] - 1) / direct_lws[0] * direct_lws[0];
    direct_gws[1] =
        (direct_gws[1] + direct_lws[1] - 1) / direct_lws[1] * direct_lws[1];
    direct_gws[2] =
        (direct_gws[2] + direct_lws[2] - 1) / direct_lws[2] * direct_lws[2];
  } else if (mode_ == ConvolutionMode::CONV_WINO) {
    if (kernelSize_ == 3) {
      kernel_data_tile =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino3_data_tile", &err);
      CheckCl(err);
      kernel_filter_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino3_filter_tile", &err);
      CheckCl(err);
      kernel_gemm_opt =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino3_gemm_opt", &err);
      CheckCl(err);
      kernel_data_untile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino3_data_untile_no_bias", &err);
      CheckCl(err);

      inputImageSize =
          inputChannel_ * (inputHeight_ / 4) * (inputWidth_ / 4) * 36;
      filterImageSize = inputChannel_ * outputChannel_ * 36;

      H = inputHeight_;
      W = inputWidth_;
      C = inputChannel_;
      K = outputChannel_;
      P = (H + 1) / 4;
      Q = (W + 1) / 4;
      X = 36;
      PQ = P * Q;

      data_lws[0] = 16;
      data_lws[1] = 16;
      data_lws[2] = 1;
      data_gws[0] = ((Q + 15) / 16) * 16;
      data_gws[1] = ((P + 15) / 16) * 16;
      data_gws[2] = C;

      int filter_BSZ = 128;
      filter_lws[0] = filter_BSZ;
      filter_lws[1] = filter_lws[2] = 1;
      filter_gws[0] = (((C * K) + filter_BSZ) / filter_BSZ) * filter_BSZ;
      filter_gws[1] = filter_gws[2] = 1;

      int gemm_TILE_M = 32;
      int gemm_TILE_N = 32;
      int gemm_BSZ = 64;
      gemm_gws[0] = X * ((P * Q + gemm_TILE_N - 1) / gemm_TILE_N) *
                    ((K + gemm_TILE_M - 1) / gemm_TILE_M) * gemm_BSZ;
      gemm_gws[1] = gemm_gws[2] = 1;
      gemm_lws[0] = gemm_BSZ;
      gemm_lws[1] = gemm_lws[2] = 1;

      int untile_BSZ = 128;
      untile_gws[0] = ((K * P * Q + untile_BSZ - 1) / untile_BSZ) * untile_BSZ;
      untile_gws[1] = untile_gws[2] = 1;
      untile_lws[0] = untile_BSZ;
      untile_lws[1] = untile_lws[2] = 1;

    } else {
      inputImageSize =
          inputChannel_ * (inputHeight_ / 4) * (inputWidth_ / 4) * 64;
      filterImageSize = inputChannel_ * outputChannel_ * 64;

      kernel_data_tile =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino5_data_tile", &err);
      CheckCl(err);
      kernel_filter_tile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino5_filter_tile", &err);
      CheckCl(err);
      kernel_gemm_opt =
          clCreateKernel(dev_->ClDnnHandle()->program, "conv_wino5_gemm_opt", &err);
      CheckCl(err);
      kernel_gemm_opt_img = clCreateKernel(dev_->ClDnnHandle()->program,
                                           "conv_wino5_gemm_opt_img", &err);
      CheckCl(err);
      kernel_data_untile = clCreateKernel(dev_->ClDnnHandle()->program,
                                          "conv_wino5_data_untile_no_bias", &err);
      CheckCl(err);

      H = inputHeight_;
      W = inputWidth_;
      C = inputChannel_;
      K = outputChannel_;
      P = (int)ceil((H - 4) / 4.0f);
      if (is_split_) P /= 2;
      Q = (int)ceil((W - 4) / 4.0f);
      X = 64;
      PQ = P * Q;

      data_lws[0] = 16;
      data_lws[1] = 16;
      data_lws[2] = 1;
      data_gws[0] = ((Q + 15) / 16) * 16;
      data_gws[1] = ((P + 15) / 16) * 16;
      data_gws[2] = C;

      int filter_BSZ = 128;
      filter_lws[0] = filter_BSZ;
      filter_lws[1] = filter_lws[2] = 1;
      filter_gws[0] = (((C * K) + filter_BSZ) / filter_BSZ) * filter_BSZ;
      filter_gws[1] = filter_gws[2] = 1;

      int gemm_TILE_M = 32;
      int gemm_TILE_N = 32;
      int gemm_BSZ = 64;
      gemm_gws[0] = X * ((P * Q + gemm_TILE_N - 1) / gemm_TILE_N) *
                    ((K + gemm_TILE_M - 1) / gemm_TILE_M) * gemm_BSZ;
      gemm_gws[1] = gemm_gws[2] = 1;
      gemm_lws[0] = gemm_BSZ;
      gemm_lws[1] = gemm_lws[2] = 1;

      int untile_BSZ = 128;
      untile_gws[0] = ((K * P * Q + untile_BSZ - 1) / untile_BSZ) * untile_BSZ;

      untile_gws[1] = untile_gws[2] = 1;
      untile_lws[0] = untile_BSZ;
      untile_lws[1] = untile_lws[2] = 1;
    }

  } else if (mode_ == ConvolutionMode::CONV_IMPLICIT) {
    imp_kernel = clCreateKernel(dev_->ClDnnHandle()->program,
                                "conv_implicit_gemm", &err);
    CheckCl(err);
    imp_kernel_img = clCreateKernel(dev_->ClDnnHandle()->program,
                                     "conv_implicit_gemm_img", &err);
    CheckCl(err);

    inputImageSize = inputChannel_ * inputHeight_ * inputWidth_;
    filterImageSize =
        inputChannel_ * outputChannel_ * kernelSize_ * kernelSize_;

    int imp_TILE_N = 32;
    int imp_TILE_M = 32;
    int imp_BSZ = 64;
    imp_gws[1] = 1;
    imp_gws[2] = 1;
    imp_lws[1] = 1;
    imp_lws[2] = 1;
    imp_lws[0] = imp_BSZ;
    imp_gws[0] =
        ((outputHeight_ * outputWidth_ + imp_TILE_N - 1) / imp_TILE_N) *
        ((outputChannel_ + imp_TILE_M - 1) / imp_TILE_M) * imp_lws[0];
  } else {
    cout << "no algorithm" << endl;
    exit(-1);
  }
}

void Convolution2d2::LoadWeight(std::ifstream &fp) {
  char *weight_H = (char *)malloc(weightSize_);
  fp.read(weight_H, weightSize_);
  weight_->Set(weight_H);
  delete weight_H;
}

void Convolution2d2::Forward(gpu::Memory *input, gpu::Memory *output) {
  cl_mem inputMem = input->Ptr();
  cl_mem outputMem = output->Ptr();
  cl_mem weightMem = weight_->Ptr();

  cl_mem inputImage, filterImage;
  double measured;

  cl_int err = 0;

  if (mode_ == ConvolutionMode::CONV_DIRECT) {
    CheckCl(clSetKernelArg(direct_kernel, 0, sizeof(cl_mem), &inputMem));
    CheckCl(clSetKernelArg(direct_kernel, 1, sizeof(cl_mem), &outputMem));
    CheckCl(clSetKernelArg(direct_kernel, 2, sizeof(cl_mem), &weightMem));
    CheckCl(clSetKernelArg(direct_kernel, 3, sizeof(cl_int), &inputChannel_));
    CheckCl(clSetKernelArg(direct_kernel, 4, sizeof(cl_int), &inputHeight_));
    CheckCl(clSetKernelArg(direct_kernel, 5, sizeof(cl_int), &inputWidth_));
    CheckCl(clSetKernelArg(direct_kernel, 6, sizeof(cl_int), &outputChannel_));
    CheckCl(clSetKernelArg(direct_kernel, 7, sizeof(cl_int), &outputHeight_));
    CheckCl(clSetKernelArg(direct_kernel, 8, sizeof(cl_int), &outputWidth_));
    CheckCl(clSetKernelArg(direct_kernel, 9, sizeof(cl_int), &kernelSize_));

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, direct_kernel, 3,
                                   NULL, direct_gws, direct_lws, 0, NULL,
                                   NULL));
  } else if (mode_ == ConvolutionMode::CONV_IMPLICIT) {
    if (inputImageSize > 134217728 * 2) {
      CheckCl(clSetKernelArg(imp_kernel, 0, sizeof(cl_mem), &inputMem));
      CheckCl(clSetKernelArg(imp_kernel, 1, sizeof(cl_mem), &outputMem));
      CheckCl(clSetKernelArg(imp_kernel, 2, sizeof(cl_mem), &weightMem));
      CheckCl(clSetKernelArg(imp_kernel, 3, sizeof(cl_int), &inputChannel_));
      CheckCl(clSetKernelArg(imp_kernel, 4, sizeof(cl_int), &outputChannel_));
      CheckCl(clSetKernelArg(imp_kernel, 5, sizeof(cl_int), &inputHeight_));
      CheckCl(clSetKernelArg(imp_kernel, 6, sizeof(cl_int), &inputWidth_));
      CheckCl(clSetKernelArg(imp_kernel, 7, sizeof(cl_int), &outputHeight_));
      CheckCl(clSetKernelArg(imp_kernel, 8, sizeof(cl_int), &outputWidth_));
      CheckCl(clSetKernelArg(imp_kernel, 9, sizeof(cl_int), &kernelSize_));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, imp_kernel, 3,
                                     NULL, imp_gws, imp_lws, 0, NULL, NULL));

    } else {
      inputImage = input->createImage2(inputImageSize);
      filterImage = weight_->createImage2(filterImageSize);

      CheckCl(clSetKernelArg(imp_kernel_img, 0, sizeof(cl_mem), &inputImage));
      CheckCl(clSetKernelArg(imp_kernel_img, 1, sizeof(cl_mem), &outputMem));
      CheckCl(clSetKernelArg(imp_kernel_img, 2, sizeof(cl_mem), &filterImage));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 3, sizeof(cl_int), &inputChannel_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 4, sizeof(cl_int), &outputChannel_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 5, sizeof(cl_int), &inputHeight_));
      CheckCl(clSetKernelArg(imp_kernel_img, 6, sizeof(cl_int), &inputWidth_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 7, sizeof(cl_int), &outputHeight_));
      CheckCl(
          clSetKernelArg(imp_kernel_img, 8, sizeof(cl_int), &outputWidth_));
      CheckCl(clSetKernelArg(imp_kernel_img, 9, sizeof(cl_int), &kernelSize_));

      CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                     imp_kernel_img, 3, NULL, imp_gws, imp_lws,
                                     0, NULL, NULL));
    }

  } else if (mode_ == ConvolutionMode::CONV_WINO) {

    // data tile
    CheckCl(clSetKernelArg(kernel_data_tile, 0, sizeof(cl_mem), &inputMem));
    CheckCl(clSetKernelArg(kernel_data_tile, 1, sizeof(cl_mem), &input_tile));
    CheckCl(clSetKernelArg(kernel_data_tile, 2, sizeof(cl_int), &C));
    CheckCl(clSetKernelArg(kernel_data_tile, 3, sizeof(cl_int), &H));
    CheckCl(clSetKernelArg(kernel_data_tile, 4, sizeof(cl_int), &W));

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_data_tile,
                                   3, NULL, data_gws, data_lws, 0, NULL, NULL));

    // filter tile
    CheckCl(clSetKernelArg(kernel_filter_tile, 0, sizeof(cl_mem), &weightMem));
    CheckCl(
        clSetKernelArg(kernel_filter_tile, 1, sizeof(cl_mem), &filter_tile));
    CheckCl(clSetKernelArg(kernel_filter_tile, 2, sizeof(cl_int), &C));
    CheckCl(clSetKernelArg(kernel_filter_tile, 3, sizeof(cl_int), &K));

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                   kernel_filter_tile, 3, NULL, filter_gws,
                                   filter_lws, 0, NULL, NULL));

    // gemm opt
    CheckCl(clSetKernelArg(kernel_gemm_opt, 0, sizeof(cl_mem), &filter_tile));
    CheckCl(clSetKernelArg(kernel_gemm_opt, 1, sizeof(cl_mem), &input_tile));
    CheckCl(clSetKernelArg(kernel_gemm_opt, 2, sizeof(cl_mem), &output_tile));
    CheckCl(clSetKernelArg(kernel_gemm_opt, 3, sizeof(cl_int), &X));
    CheckCl(clSetKernelArg(kernel_gemm_opt, 4, sizeof(cl_int), &K));
    CheckCl(clSetKernelArg(kernel_gemm_opt, 5, sizeof(cl_int), &PQ));
    CheckCl(clSetKernelArg(kernel_gemm_opt, 6, sizeof(cl_int), &C));
    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue, kernel_gemm_opt,
                                   3, NULL, gemm_gws, gemm_lws, 0, NULL, NULL));

    

    // untile
    CheckCl(
        clSetKernelArg(kernel_data_untile, 0, sizeof(cl_mem), &output_tile));
    CheckCl(clSetKernelArg(kernel_data_untile, 1, sizeof(cl_mem), &outputMem));
    CheckCl(clSetKernelArg(kernel_data_untile, 2, sizeof(cl_int), &K));
    CheckCl(clSetKernelArg(kernel_data_untile, 3, sizeof(cl_int), &H));
    CheckCl(clSetKernelArg(kernel_data_untile, 4, sizeof(cl_int), &W));

    CheckCl(clEnqueueNDRangeKernel(dev_->ClDnnHandle()->queue,
                                   kernel_data_untile, 3, NULL, untile_gws,
                                   untile_lws, 0, NULL, NULL));
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(DATA_TYPE));
}

Convolution2d2::~Convolution2d2() { delete weight_; }

}  // namespace op
}  // namespace gpu
