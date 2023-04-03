#include "upsample.h"

#include <omp.h>

namespace cpu {
namespace op {

Upsample::Upsample(Device *dev, int inputChannel, int inputHeight,
                   int inputWidth, int scale)
    : cpu::op::Operation(dev, inputChannel, inputHeight, inputWidth),
      scale_(scale) {
  outputChannel_ = inputChannel_;
  outputHeight_ = inputHeight_ * scale_;
  outputWidth_ = inputWidth_ * scale_;

  // output_ = new cpu::Memory(dev, outputChannel_ * outputHeight_ *
  // outputWidth_ * sizeof(float));
}

Upsample::~Upsample() {}

void Upsample::Forward(cpu::Memory *input, cpu::Memory *output) {
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

  float scaleX = (float)(inputWidth_ - 1) / (outputWidth_ - 1);
  float scaleY = (float)(inputHeight_ - 1) / (outputHeight_ - 1);

#pragma omp parallel for
  for (int k = 0; k < outputChannel_; ++k) {
    float ratio, ow_, oh_;
    int iw, ih, idx;

    //// 1) linear interpolation in the x-direction
    for (int oh = 0; oh < outputHeight_; oh += outputHeight_ - 1) {
      // ih
      ih = oh * scaleY;

      // out(oh, 0) <== in(ih, 0)
      O[(k * outputHeight_ + oh) * outputWidth_] =
          I[(k * inputHeight_ + ih) * inputWidth_];

      for (int ow = 1; ow < outputWidth_ - 1; ++ow) {
        // iw, ratio
        ow_ = ow * scaleX;  // output coord in input
        iw = floor(ow_);    // nearest input coord
        ratio = (ow_ - iw);

        // output
        idx = (k * inputHeight_ + ih) * inputWidth_ + iw;
        O[(k * outputHeight_ + oh) * outputWidth_ + ow] =
            I[idx] * (1 - ratio) + I[idx + 1] * ratio;
      }

      // out(oh, outputWidth_-1) <== in(ih, inputWidth_-1)
      O[(k * outputHeight_ + oh) * outputWidth_ + outputWidth_ - 1] =
          I[(k * inputHeight_ + ih) * inputWidth_ + inputWidth_ - 1];
    }

    //// 2) linear interpolation in the y-direction
    float valL, valR, valT, valB;
    float ratioX, ratioY;
    for (int oh = 1; oh < outputHeight_ - 1; ++oh) {
      // ih, ratioY
      oh_ = oh * scaleY;
      ih = floor(oh_);
      ratioY = (oh_ - ih);

      // out(oh, 0) <== in(ih, 0)
      idx = (k * inputHeight_ + ih) * inputWidth_;
      valT = I[idx];
      idx += inputWidth_;
      valB = I[idx];
      O[(k * outputHeight_ + oh) * outputWidth_] =
          valT * (1 - ratioY) + valB * ratioY;

      for (int ow = 1; ow < outputWidth_ - 1; ++ow) {
        // iw, ratioX
        ow_ = ow * scaleX;  // output coord in input
        iw = floor(ow_);    // nearest input coord
        ratioX = (ow_ - iw);

        // (ih, iw)
        idx = (k * inputHeight_ + ih) * inputWidth_ + iw;
        valL = I[idx];
        // (ih, iw+1)
        idx++;
        valR = I[idx];
        valT = valL * (1 - ratioX) + valR * ratioX;

        // (ih+1, iw)
        idx = (k * inputHeight_ + ih + 1) * inputWidth_ + iw;
        valL = I[idx];
        // (ih+1, iw+1)
        idx++;
        valR = I[idx];
        valB = valL * (1 - ratioX) + valR * ratioX;

        // output
        O[(k * outputHeight_ + oh) * outputWidth_ + ow] =
            valT * (1 - ratioY) + valB * ratioY;
      }

      // out(oh, 0) <== in(ih, 0)
      idx = (k * inputHeight_ + ih) * inputWidth_ + inputWidth_ - 1;
      valT = I[idx];
      idx += inputWidth_;
      valB = I[idx];
      O[(k * outputHeight_ + oh) * outputWidth_ + outputWidth_ - 1] =
          valT * (1 - ratioY) + valB * ratioY;
    }
  }

  output->SetUsedSize(outputChannel_ * outputHeight_ * outputWidth_ *
                      sizeof(float));
}

}  // namespace op
}  // namespace cpu
