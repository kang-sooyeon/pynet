#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define EPSILON 0.00001

#define WARP_SIZE 8
#define TILE_M 32
#define TILE_K 16
#define TILE_N 32
#define PT_M 4
#define PT_N 4
#define BSZ 64

#define TP_M (TILE_M / PT_M)
#define TP_N (TILE_N / PT_N)
#if (TILE_M % PT_M != 0) || (TILE_N % PT_N != 0) || ((TP_M * TP_N) != BSZ)
#error "Invalid distribution size: Num threads"
#endif

#define WARP_CNT (BSZ / WARP_SIZE)
#define SAFE_LOAD(MAT, SY, SX, y, x)                                           \
  (((x) < (SX)) ? (((y) < (SY)) ? ((MAT)[(y) * (SX) + (x)]) : 0.h) : 0.h)
#define SAFE_LOAD_HALF(MAT, SY, SX, y, x)                                      \
  (((x) < (SX)) ? (((y) < (SY)) ? (vload_half((y) * (SX) + (x), (MAT))) : 0.h) \
                : 0.h)
#define SAFE_STORE(MAT, SY, SX, y, x, v)                                       \
  if (((x) < (SX)) && ((y) < (SY)))                                            \
    ((MAT)[(y) * (SX) + (x)] = (v));
#define SAFE_STORE_HALF(MAT, SY, SX, y, x, v)                                  \
  if (((x) < (SX)) && ((y) < (SY)))                                            \
  (vstore_half((v), (y) * (SX) + (x), (MAT)))
#define SAFE_LOAD4(MAT, SY, SX, y, x)                                          \
  ((((x)) < (SX))                                                              \
       ? (((y) < (SY)) ? (((__global float4 *)(MAT))[((y) * (SX) + (x)) / 4])  \
                       : zero)                                                 \
       : zero)

#define SAFE_IMAGE_LOAD(IMG, SY, SX, Y, X, C)                                  \
  (((X) < (SX))                                                                \
       ? (((Y) < (SY)) ? (read_imageh((IMG), (Y) * (SX) + (X) + (C)).x) : 0)   \
       : 0)

#define SAFE_IMAGE_LOAD_HALF(IMG, SY, SX, Y, X, C)                             \
  (((X) < (SX))                                                                \
       ? (((Y) < (SY))                                                         \
              ? ((float)(read_imageh((IMG), (Y) * (SX) + (X) + (C)).x))        \
              : 0)                                                             \
       : 0)

#define SAFE_IMAGE_LOAD_HALF2(IMG, SY, SX, Y, X)                               \
  (((X) < (SX)) ? (((Y) < (SY)) ? IMG : 0) : 0)

#define PT_N_OPT 2
#define TP_N_OPT (TILE_N / PT_N_OPT / 2)

#define SAFE_VEC2_STORE(MAT, SY, SX, y, x, v)                                  \
  if (((x) < (SX)) && ((y) < (SY)))                                            \
    vstore2((v), ((y) * (SX) + (x)) >> 1, (MAT));

#define SAFE_VEC2_STORE_HALF(MAT, SY, SX, y, x, v)                             \
  if (((x) < (SX)) && ((y) < (SY)))                                            \
    vstore_half2((v), ((y) * (SX) + (x)) >> 1, (MAT));

#define SAFE_IMAGE_LOAD_VEC2(IMG, SY, SX, Y, X, C)                             \
  (((X) < (SX))                                                                \
       ? (((Y) < (SY))                                                         \
              ? ((read_imagef((IMG), ((Y) * (SX) + (X) + (C)) >> 1).xy))       \
              : 0)                                                             \
       : 0)

#define SAFE_IMAGE_LOAD_VEC2_HALF(IMG, SY, SX, Y, X, C)                        \
  (((X) < (SX))                                                                \
       ? (((Y) < (SY))                                                         \
              ? ((read_imageh((IMG), ((Y) * (SX) + (X) + (C)) >> 1).xy))       \
              : 0)                                                             \
       : 0)

__kernel void convert_to_fp16(__global float *in, __global half *out, int C,
                              int H, int W) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  int c = get_global_id(2);

  if (w >= W || h >= H || c >= C)
    return;

  int idx = (c * H + h) * W + w;
  float val = in[idx];
  vstore_half(val, idx, out);
}

__kernel void convert_from_fp16(__global half *in, __global float *out, int C,
                                int H, int W) {

  int w = get_global_id(0);
  int h = get_global_id(1);
  int c = get_global_id(2);

  if (w >= W || h >= H || c >= C)
    return;

  int idx = (c * H + h) * W + w;
  float val = vload_half(idx, in);
  out[idx] = val;
}

__kernel void extractBayerChannel(__global size_t *input, __global half *output,
                                  int inputChannel_, int inputHeight_,
                                  int inputWidth_, int outputChannel_,
                                  int outputHeight_, int outputWidth_) {
  int ih = get_global_id(1);
  int iw = get_global_id(0);

  if (ih >= inputHeight_ || iw >= inputWidth_) {
    return;
  }

  bool isOddH = ih & 0x1;
  bool isOddW = iw & 0x1;

  int oh = ih / 2;
  int ow = iw / 2;

  half norm = 1020.h;

  // channel order : B, GB, R, GR
  if (isOddH) {
    if (isOddW) {
      // B channel
      output[oh * outputWidth_ + ow] =
          (half)(input[ih * inputWidth_ + iw]) / norm;
    } else {
      // GR channel
      output[(3 * outputHeight_ + oh) * outputWidth_ + ow] =
          (half)(input[ih * inputWidth_ + iw]) / norm;
    }
  } else {
    if (isOddW) {
      // GB channel
      output[(outputHeight_ + oh) * outputWidth_ + ow] =
          (half)(input[ih * inputWidth_ + iw]) / norm;

    } else {
      // R channel
      output[(2 * outputHeight_ + oh) * outputWidth_ + ow] =
          (half)(input[ih * inputWidth_ + iw]) / norm;
    }
  }
}

__kernel void maxPool2d_img(__read_only image1d_buffer_t input,
                            __global half *output, int inputChannel_,
                            int inputHeight_, int inputWidth_,
                            int outputChannel_, int outputHeight_,
                            int outputWidth_, int kernelSize_, int stride_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);
  int idx;
  half2 reg;

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_) {
    return;
  }

  half max = (half)(-FLT_MAX);
  half val;
  int ih, iw;
  for (int r = 0; r < kernelSize_; r++) {
    for (int s = 0; s < kernelSize_; s++) {
      ih = stride_ * oh + r;
      iw = stride_ * ow + s;
      idx = (k * inputHeight_ + ih) * inputWidth_ + iw;
      reg = read_imageh(input, idx >> 1).xy;
      val = (idx % 2 == 0) ? reg.x : reg.y;
      if (max < val)
        max = val;
    }
  }
  output[(k * outputHeight_ + oh) * outputWidth_ + ow] = max;
}

__kernel void maxPool2d(__global half *input, __global half *output,
                        int inputChannel_, int inputHeight_, int inputWidth_,
                        int outputChannel_, int outputHeight_, int outputWidth_,
                        int kernelSize_, int stride_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_) {
    return;
  }

  half max = (half)(-FLT_MAX);
  half val;
  int ih, iw;
  for (int r = 0; r < kernelSize_; r++) {
    for (int s = 0; s < kernelSize_; s++) {
      ih = stride_ * oh + r;
      iw = stride_ * ow + s;
      val = input[(k * inputHeight_ + ih) * inputWidth_ + iw];
      if (max < val)
        max = val;
    }
  }
  output[(k * outputHeight_ + oh) * outputWidth_ + ow] = max;
}

__kernel void reduction_float_img(__read_only image1d_buffer_t input,
                                  __global float *groupSum,
                                  __local float *localSum, int HW, int groupNum,
                                  int inputChannel_, int offset) {

  int gi = get_global_id(0);
  int li = get_local_id(0);
  int idx;

  for (int c = 0; c < inputChannel_; c++) {
    idx = c * HW + gi;
    float2 reg = read_imagef(input, idx >> 1).xy;
    localSum[li] = (gi < HW) ? (idx % 2 == 0 ? reg.x : reg.y) : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = get_local_size(0) / 2; p >= 1; p = (p >> 1)) {
      if (li < p)
        localSum[li] += localSum[li + p];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // move result to groupSum
    if (li == 0) {
      // groupSum[get_group_id(0)] = localSum[0];
      groupSum[get_group_id(0) + (c + offset) * groupNum] = localSum[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel void reduction_float(__global float *input, __global float *groupSum,
                              __local float *localSum, int HW, int groupNum,
                              int inputChannel_, int offset) {

  int gi = get_global_id(0);
  int li = get_local_id(0);

  for (int c = 0; c < inputChannel_; c++) {
    localSum[li] = (gi < HW) ? input[gi + c * HW] : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = get_local_size(0) / 2; p >= 1; p = (p >> 1)) {
      if (li < p)
        localSum[li] += localSum[li + p];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // move result to groupSum
    if (li == 0) {
      // groupSum[get_group_id(0)] = localSum[0];
      groupSum[get_group_id(0) + (c + offset) * groupNum] = localSum[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel void reduction_img(__read_only image1d_buffer_t input,
                            __global float *groupSum, __local float *localSum,
                            int HW, int groupNum, int inputChannel_) {
  int gi = get_global_id(0);
  int li = get_local_id(0);
  int idx;
  half2 reg;

  for (int c = 0; c < inputChannel_; c++) {
    idx = gi + c * HW;

    reg = read_imageh(input, idx >> 1).xy;

    localSum[li] =
        (gi < HW) ? (idx % 2 == 0 ? (float)reg.x : (float)reg.y) : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = get_local_size(0) / 2; p >= 1; p = (p >> 1)) {
      if (li < p)
        localSum[li] += localSum[li + p];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // move result to groupSum
    if (li == 0) {
      groupSum[get_group_id(0) + c * groupNum] = localSum[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel void reduction(__global half *input, __global float *groupSum,
                        __local float *localSum, int HW, int groupNum,
                        int inputChannel_) {
  int gi = get_global_id(0);
  int li = get_local_id(0);

  for (int c = 0; c < inputChannel_; c++) {
    localSum[li] = (gi < HW) ? vload_half(gi + c * HW, input) : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = get_local_size(0) / 2; p >= 1; p = (p >> 1)) {
      if (li < p)
        localSum[li] += localSum[li + p];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // move result to groupSum
    if (li == 0) {
      // groupSum[get_group_id(0)] = localSum[0];
      groupSum[get_group_id(0) + c * groupNum] = localSum[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel void mean_img(__read_only image1d_buffer_t groupSum, int elems,
                       int groupNum, int HW, __global float *mean) {

  int i = get_global_id(0);
  if (i >= elems)
    return;

  int idx;

  int k = i % groupNum; // i : 전체 채널의 그룹갯수, groupNum : 한 채널당 그룹
                        // 갯수 // k == 한 채널에서 그룹 index
  if (k == 0) {
    int c = i / groupNum;
    float x = 0;
    for (int n = 0; n < groupNum; n++) {
      idx = n + c * groupNum;
      float2 reg = read_imagef(groupSum, idx >> 1).xy;
      x += (idx % 2 == 0 ? reg.x : reg.y);
    }
    mean[c] = x / HW;
  }
}

__kernel void mean(__global float *groupSum, int elems, int groupNum, int HW,
                   __global float *mean) {

  int i = get_global_id(0);
  if (i >= elems)
    return;

  int k = i % groupNum; // i : 전체 채널의 그룹갯수, groupNum : 한 채널당 그룹
                        // 갯수 // k == 한 채널에서 그룹 index
  if (k == 0) {
    int c = i / groupNum;
    float x = 0;
    for (int n = 0; n < groupNum; n++) {
      x += groupSum[n + c * groupNum];
    }
    mean[c] = x / HW;
  }
}

__kernel void subSquare_img(__read_only image1d_buffer_t input,
                            __global float *output, __global float *mean,
                            int channel, int height, int width, int offset) {
  int i = get_global_id(0);
  int idx;
  if (i >= channel * height * width)
    return;

  int HW = height * width;
  int c = (i / HW) + offset;

  idx = (height * width * offset) + i;
  float2 reg = read_imagef(input, idx >> 1).xy;

  float x = (idx % 2 == 0 ? reg.x : reg.y) - mean[c];
  output[i] = x * x;
}

__kernel void subSquare(__global half *input, __global float *output,
                        __global float *mean, int channel, int height,
                        int width, int offset) {
  int i = get_global_id(0);
  if (i >= channel * height * width)
    return;

  int HW = height * width;
  int c = (i / HW) + offset;

  float x = vload_half((height * width * offset) + i, input) - mean[c];
  output[i] = x * x;
}

__kernel void std_img(__read_only image1d_buffer_t groupSum, int elems,
                      int groupNum, int HW, __global float *std, int offset) {

  int i = get_global_id(0);
  if (i >= elems)
    return;

  int idx;
  int k = i % groupNum;
  if (k == 0) {
    int c = i / groupNum;
    float x = 0;
    for (int n = 0; n < groupNum; n++) {
      idx = n + (c + offset) * groupNum;
      float2 reg = read_imagef(groupSum, idx >> 1).xy;
      x += (idx % 2 == 0 ? reg.x : reg.y);
    }

    x /= HW;
    x += EPSILON;
    x = sqrt(x);
    std[c + offset] = x;
  }
}

__kernel void std(__global float *groupSum, int elems, int groupNum, int HW,
                  __global float *std, int offset) {

  int i = get_global_id(0);
  if (i >= elems)
    return;

  int k = i % groupNum;
  if (k == 0) {
    int c = i / groupNum;
    float x = 0;
    for (int n = 0; n < groupNum; n++) {
      x += groupSum[n + (c + offset) * groupNum];
    }

    x /= HW;
    x += EPSILON;
    x = sqrt(x);
    std[c + offset] = x;
  }
}

__kernel void instanceNorm2d(__global half *input, __global half *weight,
                             __global half *bias, __global float *mean,
                             __global float *std, int channel, int height,
                             int width, int reluOn) {

  int i = get_global_id(0);
  if (i >= channel * height * width)
    return;

  int HW = height * width;
  int c = i / HW;

  float ret =
      vload_half(c, weight) * ((vload_half(i, input) - mean[c]) / std[c]) +
      vload_half(c, bias);

  if (ret > 0 || !reluOn)
    vstore_half(ret, i, input);
  else
    vstore_half(ret * 0.2h, i, input);
}

__kernel void add(__global half *input1, __global half *input2, int num) {
  int i = get_global_id(0);

  if (i >= num)
    return;

  input2[i] = input1[i] + input2[i];
}

__kernel void split(__global half *input, __global half *output, int n) {
  int i = get_global_id(0);

  if (i >= n)
    return;

  output[i] = input[i + n];
}

__kernel void leakyReLU_img(__read_only image1d_buffer_t input,
                            __global half2 *output, int outputChannel_,
                            int outputHeight_, int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  int idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
  half2 val = read_imageh(input, idx).xy;

  val.x = (val.x > 0) ? val.x : val.x * 0.2h;
  val.y = (val.y > 0) ? val.y : val.y * 0.2h;

  output[idx] = val;
}

__kernel void leakyReLU(__global half *input, __global half *output,
                        int outputChannel_, int outputHeight_,
                        int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  int idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
  half val = input[idx];

  if (val > 0)
    output[idx] = val;
  else
    output[idx] = val * 0.2h;
}

__kernel void convolution2d(__global half *input, __global half *output,
                            __global half *weight, int inputChannel_,
                            int inputHeight_, int inputWidth_,
                            int outputChannel_, int outputHeight_,
                            int outputWidth_, int kernelSize_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_) {
    return;
  }

  half sum = 0.h;
  for (int c = 0; c < inputChannel_; c++) {
    for (int r = 0; r < kernelSize_; r++) {
      int ih = oh + r;
      if (ih < 0 || ih >= inputHeight_)
        continue;
      for (int s = 0; s < kernelSize_; s++) {
        int iw = ow + s;
        if (iw < 0 || iw >= inputWidth_)
          continue;
        sum +=
            input[(c * inputHeight_ + ih) * inputWidth_ + iw] *
            weight[((k * inputChannel_ + c) * kernelSize_ + r) * kernelSize_ +
                   s];
      }
    }
  }
  output[(k * outputHeight_ + oh) * outputWidth_ + ow] = sum;
}

__kernel void convolution2d2(__global half *input, __global half *output,
                             __global half *weight, __global half *bias,
                             int inputChannel_, int inputHeight_,
                             int inputWidth_, int outputChannel_,
                             int outputHeight_, int outputWidth_,
                             int kernelSize_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_) {
    return;
  }

  half sum = 0.h;
  for (int c = 0; c < inputChannel_; c++) {
    for (int r = 0; r < kernelSize_; r++) {
      int ih = oh + r;
      if (ih < 0 || ih >= inputHeight_)
        continue;
      for (int s = 0; s < kernelSize_; s++) {
        int iw = ow + s;
        if (iw < 0 || iw >= inputWidth_)
          continue;
        sum +=
            input[(c * inputHeight_ + ih) * inputWidth_ + iw] *
            weight[((k * inputChannel_ + c) * kernelSize_ + r) * kernelSize_ +
                   s];
      }
    }
  }
  sum += bias[k];
  output[(k * outputHeight_ + oh) * outputWidth_ + ow] = sum;
}

__kernel void addBias(__global half *input, __global half *bias,
                      int inputChannel_, int inputHeight_, int inputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= inputChannel_ || oh >= inputHeight_ || ow >= inputWidth_) {
    return;
  }

  input[(k * inputHeight_ + oh) * inputWidth_ + ow] += bias[k];
}

__kernel void upsample_(__global half *input, __global half *output,
                        int inputChannel_, int inputHeight_, int inputWidth_,
                        int outputChannel_, int outputHeight_, int outputWidth_,
                        int scale_

) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  float scaleX = (float)(inputWidth_ - 1) / (outputWidth_ - 1);
  float scaleY = (float)(inputHeight_ - 1) / (outputHeight_ - 1);

  float ow_, oh_;
  int iw, ih, idx;

  if ((oh == 0 && ow == 0) || (oh == 0 && ow == outputWidth_ - 1) ||
      (oh == outputHeight_ - 1 && ow == 0) ||
      (oh == outputHeight_ - 1 && ow == outputWidth_ - 1)) {
    iw = ow * scaleX;
    ih = oh * scaleY;

    float ret = vload_half((k * inputHeight_ + ih) * inputWidth_ + iw, input);
    vstore_half(ret, (k * outputHeight_ + oh) * outputWidth_ + ow, output);
    return;
  }

  // iw, ratioX
  ow_ = ow * scaleX;
  iw = floor(ow_);
  float ratioX = (ow_ - iw);

  // ih, ratioY
  oh_ = oh * scaleY;
  ih = floor(oh_);
  float ratioY = (oh_ - ih);

  if (oh == 0 || oh == outputHeight_ - 1) {
    idx = (k * inputHeight_ + ih) * inputWidth_ + iw;
    float ret = vload_half(idx, input) * (1 - ratioX) +
                vload_half(idx + 1, input) * ratioX;
    vstore_half(ret, (k * outputHeight_ + oh) * outputWidth_ + ow, output);
    return;
  }

  if (ow == 0 || ow == outputWidth_ - 1) {
    idx = (k * inputHeight_ + ih) * inputWidth_ + iw;
    float ret = vload_half(idx, input) * (1 - ratioY) +
                vload_half(idx + inputWidth_, input) * ratioY;
    vstore_half(ret, (k * outputHeight_ + oh) * outputWidth_ + ow, output);
    return;
  }

  // val Top(ih) : val Left, Right
  idx = (k * inputHeight_ + ih) * inputWidth_ + iw;
  float valL = vload_half(idx, input);
  float valR = vload_half(idx + 1, input);
  float valT = valL * (1 - ratioX) + valR * ratioX;

  // val Top(ih+1) : val Left, Right
  idx += inputWidth_;
  valL = vload_half(idx, input);
  valR = vload_half(idx + 1, input);
  float valB = valL * (1 - ratioX) + valR * ratioX;

  float ret = valT * (1 - ratioY) + valB * ratioY;
  vstore_half(ret, (k * outputHeight_ + oh) * outputWidth_ + ow, output);
}

__kernel void cat1_img(__read_only image1d_buffer_t input1,
                       __read_only image1d_buffer_t input2,
                       __global half *output, int inputChannel1_,
                       int inputHeight1_, int inputWidth1_, int inputChannel2_,
                       int inputHeight2_, int inputWidth2_, int outputChannel_,
                       int outputHeight_, int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);
  int idx1, idx2;
  half2 regA, regB;

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  if (k < inputChannel1_) {
    idx1 = (k * inputHeight1_ + oh) * inputWidth1_ + ow;
    regA = read_imageh(input1, idx1 >> 1).xy;

    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        idx1 % 2 == 0 ? regA.x : regA.y;
  } else {
    idx2 = ((k - inputChannel1_) * inputHeight2_ + oh) * inputWidth2_ + ow;
    regB = read_imageh(input2, idx2 >> 1).xy;

    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        idx2 % 2 == 0 ? regB.x : regB.y;
  }
}

__kernel void cat1(__global half *input1, __global half *input2,
                   __global half *output, int inputChannel1_, int inputHeight1_,
                   int inputWidth1_, int inputChannel2_, int inputHeight2_,
                   int inputWidth2_, int outputChannel_, int outputHeight_,
                   int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  if (k < inputChannel1_) {
    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        input1[(k * inputHeight1_ + oh) * inputWidth1_ + ow];
  } else {
    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        input2[((k - inputChannel1_) * inputHeight2_ + oh) * inputWidth2_ + ow];
  }
}

__kernel void cat2(__global half *input1, __global half *input2,
                   __global half *output, int inputChannel1_, int inputHeight1_,
                   int inputWidth1_, int inputChannel2_, int inputHeight2_,
                   int inputWidth2_, int outputChannel_, int outputHeight_,
                   int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  if (oh < inputHeight1_) {
    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        input1[(k * inputHeight1_ + oh) * inputWidth1_ + ow];
  } else {
    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        input2[(k * inputHeight2_ + (oh - inputHeight1_)) * inputWidth2_ + ow];
  }
}

__kernel void cat3(__global half *input1, __global half *input2,
                   __global half *output, int inputChannel1_, int inputHeight1_,
                   int inputWidth1_, int inputChannel2_, int inputHeight2_,
                   int inputWidth2_, int outputChannel_, int outputHeight_,
                   int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  if (ow < inputWidth1_) {
    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        input1[(k * inputHeight1_ + oh) * inputWidth1_ + ow];
  } else {
    output[(k * outputHeight_ + oh) * outputWidth_ + ow] =
        input2[(k * inputHeight2_ + oh) * inputWidth2_ + (ow - inputWidth1_)];
  }
}

__kernel void reflectionPad2d_img(__read_only image1d_buffer_t input,
                                  __global half *output, int inputChannel_,
                                  int inputHeight_, int inputWidth_,
                                  int outputChannel_, int outputHeight_,
                                  int outputWidth_, int pad0, int pad1,
                                  int pad2, int pad3) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_) {
    return;
  }

  int input_idx;
  int output_idx;
  half2 reg;

  if (oh < pad2 || oh > (pad2 + (inputHeight_ - 1)) || ow < pad0 ||
      ow > (pad0 + (inputWidth_ - 1))) {
    if (oh < pad2) {
      if (ow < pad0) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx =
            (k * inputHeight_ + (pad2 - oh)) * inputWidth_ + (pad0 - ow);
      } else if (ow > (pad0 + (inputWidth_ - 1))) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (pad2 - oh)) * inputWidth_ +
                    (2 * inputWidth_ + pad0 - ow - 2);
      } else {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx =
            (k * inputHeight_ + (pad2 - oh)) * inputWidth_ + (ow - pad0);
      }
    } else if (oh > (pad2 + (inputHeight_ - 1))) {
      if (ow < pad0) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (2 * inputHeight_ + pad2 - oh - 2)) *
                        inputWidth_ +
                    (pad0 - ow);
      } else if (ow > (pad0 + (inputWidth_ - 1))) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (2 * inputHeight_ + pad2 - oh - 2)) *
                        inputWidth_ +
                    (2 * inputWidth_ + pad0 - ow - 2);
      } else {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (2 * inputHeight_ + pad2 - oh - 2)) *
                        inputWidth_ +
                    (ow - pad0);
      }
    } else {
      if (ow < pad0) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx =
            (k * inputHeight_ + (oh - pad2)) * inputWidth_ + (pad0 - ow);
      } else {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (oh - pad2)) * inputWidth_ +
                    (2 * inputWidth_ + pad0 - ow - 2);
      }
    }
  } else {
    output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
    input_idx = (k * inputHeight_ + (oh - pad2)) * inputWidth_ + (ow - pad0);
  }

  reg = read_imageh(input, input_idx >> 1).xy;

  output[output_idx] = input_idx % 2 == 0 ? reg.x : reg.y;
}

// area position
// 0 |  2  | 1
// 6 | pad | 7
// 3 |  5  | 4

__kernel void reflectionPad2d(__global half *input, __global half *output,
                              int inputChannel_, int inputHeight_,
                              int inputWidth_, int outputChannel_,
                              int outputHeight_, int outputWidth_, int pad0,
                              int pad1, int pad2, int pad3) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_) {
    return;
  }

  int input_idx;
  int output_idx;

  if (oh < pad2 || oh > (pad2 + (inputHeight_ - 1)) || ow < pad0 ||
      ow > (pad0 + (inputWidth_ - 1))) {
    if (oh < pad2) {
      if (ow < pad0) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx =
            (k * inputHeight_ + (pad2 - oh)) * inputWidth_ + (pad0 - ow);
      } else if (ow > (pad0 + (inputWidth_ - 1))) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (pad2 - oh)) * inputWidth_ +
                    (2 * inputWidth_ + pad0 - ow - 2);
      } else {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx =
            (k * inputHeight_ + (pad2 - oh)) * inputWidth_ + (ow - pad0);
      }
    } else if (oh > (pad2 + (inputHeight_ - 1))) {
      if (ow < pad0) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (2 * inputHeight_ + pad2 - oh - 2)) *
                        inputWidth_ +
                    (pad0 - ow);
      } else if (ow > (pad0 + (inputWidth_ - 1))) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (2 * inputHeight_ + pad2 - oh - 2)) *
                        inputWidth_ +
                    (2 * inputWidth_ + pad0 - ow - 2);
      } else {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (2 * inputHeight_ + pad2 - oh - 2)) *
                        inputWidth_ +
                    (ow - pad0);
      }
    } else {
      if (ow < pad0) {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx =
            (k * inputHeight_ + (oh - pad2)) * inputWidth_ + (pad0 - ow);
      } else {
        output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
        input_idx = (k * inputHeight_ + (oh - pad2)) * inputWidth_ +
                    (2 * inputWidth_ + pad0 - ow - 2);
      }
    }
  } else {
    output_idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
    input_idx = (k * inputHeight_ + (oh - pad2)) * inputWidth_ + (ow - pad0);
  }

  output[output_idx] = input[input_idx];
}

__kernel void sigmoid_img(__read_only image1d_buffer_t input,
                          __global half2 *output, int outputChannel_,
                          int outputHeight_, int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  int idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
  output[idx] = 1.h / (1.h + exp(-read_imageh(input, idx).xy));
}

__kernel void sigmoid(__global half *input, __global half *output,
                      int outputChannel_, int outputHeight_, int outputWidth_) {
  int k = get_global_id(2);
  int oh = get_global_id(1);
  int ow = get_global_id(0);

  if (k >= outputChannel_ || oh >= outputHeight_ || ow >= outputWidth_)
    return;

  int idx = (k * outputHeight_ + oh) * outputWidth_ + ow;
  output[idx] = 1.h / (1.h + exp(-input[idx]));
}

////////////////////////////////////////////////////////////// advanced
/// convolution

__kernel void conv_wino3_data_tile(__global half *inputs,
                                   __global half *outputs, int C, int H,
                                   int W) {
  int PH = H;
  int PW = W;
  int TP = (H + 1) / 4;
  int TQ = (W + 1) / 4;
  int c = get_global_id(2);
  int tp = get_global_id(1);
  int tq = get_global_id(0);
  if (tp >= TP || tq >= TQ)
    return;
  int h = tp * 4;
  int w = tq * 4;
  int lid_tp = get_local_id(1);
  int lid_tq = get_local_id(0);
  int lid_h = lid_tp * 4;
  int lid_w = lid_tq;

  __local half4 v[66][17];

  half4 v1;

  inputs += (c * PH + h) * PW + w; // jump to tile offset (input)

  // boundary tiles
  if (lid_tp == 15 || lid_tq == 15 || tp == TP - 1 || tq == TQ - 1) {

    // 6x8 load
    for (int i = 0; i < 6; i++) {
      v1.x = ((h + i < PH) && (w + 0 < PW)) ? inputs[i * PW + 0] : 0.h;
      v1.y = ((h + i < PH) && (w + 1 < PW)) ? inputs[i * PW + 1] : 0.h;
      v1.z = ((h + i < PH) && (w + 2 < PW)) ? inputs[i * PW + 2] : 0.h;
      v1.w = ((h + i < PH) && (w + 3 < PW)) ? inputs[i * PW + 3] : 0.h;
      v[lid_h + i][lid_w] = v1;
      v1.x = ((h + i < PH) && (w + 4 < PW)) ? inputs[i * PW + 4] : 0.h;
      v1.y = ((h + i < PH) && (w + 5 < PW)) ? inputs[i * PW + 5] : 0.h;
      v1.z = ((h + i < PH) && (w + 6 < PW)) ? inputs[i * PW + 6] : 0.h;
      v1.w = ((h + i < PH) && (w + 7 < PW)) ? inputs[i * PW + 7] : 0.h;
      v[lid_h + i][lid_w + 1] = v1;
    }

  } else {

    // 4x4 load
    for (int i = 0; i < 4; i++) {
      v1 = vload4(0, inputs);
      v[lid_h + i][lid_w] = v1;
      inputs += PW;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  outputs += (c * TP + tp) * TQ + tq; // jump to tile offset to save (output)
  int K = C * TP * TQ;

  half TV0, TV1, TV2, TV3, TV4, TV5;
  half V0, V1, V2, V3, V4, V5;

  for (int j = 0; j < 6; j++) {

    switch (j) {
    case 0:
      TV0 = 4.h * v[lid_h + 0][lid_w].x - 5.h * v[lid_h + 2][lid_w].x +
            1.h * v[lid_h + 4][lid_w].x;
      TV1 = 4 * v[lid_h + 0][lid_w].y - 5.h * v[lid_h + 2][lid_w].y +
            1.h * v[lid_h + 4][lid_w].y;
      TV2 = 4.h * v[lid_h + 0][lid_w].z - 5.h * v[lid_h + 2][lid_w].z +
            1.h * v[lid_h + 4][lid_w].z;
      TV3 = 4.h * v[lid_h + 0][lid_w].w - 5.h * v[lid_h + 2][lid_w].w +
            1.h * v[lid_h + 4][lid_w].w;
      TV4 = 4.h * v[lid_h + 0][lid_w + 1].x - 5.h * v[lid_h + 2][lid_w + 1].x +
            1.h * v[lid_h + 4][lid_w + 1].x;
      TV5 = 4.h * v[lid_h + 0][lid_w + 1].y - 5.h * v[lid_h + 2][lid_w + 1].y +
            1.h * v[lid_h + 4][lid_w + 1].y;
      break;

    case 1:
      TV0 = -4.h * v[lid_h + 1][lid_w + 0].x - 4.h * v[lid_h + 2][lid_w + 0].x +
            1.h * v[lid_h + 3][lid_w + 0].x + 1.h * v[lid_h + 4][lid_w + 0].x;
      TV1 = -4.h * v[lid_h + 1][lid_w + 0].y - 4.h * v[lid_h + 2][lid_w + 0].y +
            1.h * v[lid_h + 3][lid_w + 0].y + 1.h * v[lid_h + 4][lid_w + 0].y;
      TV2 = -4.h * v[lid_h + 1][lid_w + 0].z - 4.h * v[lid_h + 2][lid_w + 0].z +
            1.h * v[lid_h + 3][lid_w + 0].z + 1.h * v[lid_h + 4][lid_w + 0].z;
      TV3 = -4.h * v[lid_h + 1][lid_w + 0].w - 4.h * v[lid_h + 2][lid_w + 0].w +
            1.h * v[lid_h + 3][lid_w + 0].w + 1.h * v[lid_h + 4][lid_w + 0].w;
      TV4 = -4.h * v[lid_h + 1][lid_w + 1].x - 4.h * v[lid_h + 2][lid_w + 1].x +
            1.h * v[lid_h + 3][lid_w + 1].x + 1.h * v[lid_h + 4][lid_w + 1].x;
      TV5 = -4.h * v[lid_h + 1][lid_w + 1].y - 4.h * v[lid_h + 2][lid_w + 1].y +
            1.h * v[lid_h + 3][lid_w + 1].y + 1.h * v[lid_h + 4][lid_w + 1].y;
      break;

    case 2:
      TV0 = 4.h * v[lid_h + 1][lid_w + 0].x - 4.h * v[lid_h + 2][lid_w + 0].x -
            1.h * v[lid_h + 3][lid_w + 0].x + 1.h * v[lid_h + 4][lid_w + 0].x;
      TV1 = 4.h * v[lid_h + 1][lid_w + 0].y - 4.h * v[lid_h + 2][lid_w + 0].y -
            1.h * v[lid_h + 3][lid_w + 0].y + 1.h * v[lid_h + 4][lid_w + 0].y;
      TV2 = 4.h * v[lid_h + 1][lid_w + 0].z - 4.h * v[lid_h + 2][lid_w + 0].z -
            1.h * v[lid_h + 3][lid_w + 0].z + 1.h * v[lid_h + 4][lid_w + 0].z;
      TV3 = 4.h * v[lid_h + 1][lid_w + 0].w - 4.h * v[lid_h + 2][lid_w + 0].w -
            1.h * v[lid_h + 3][lid_w + 0].w + 1.h * v[lid_h + 4][lid_w + 0].w;
      TV4 = 4.h * v[lid_h + 1][lid_w + 1].x - 4.h * v[lid_h + 2][lid_w + 1].x -
            1.h * v[lid_h + 3][lid_w + 1].x + 1.h * v[lid_h + 4][lid_w + 1].x;
      TV5 = 4.h * v[lid_h + 1][lid_w + 1].y - 4.h * v[lid_h + 2][lid_w + 1].y -
            1.h * v[lid_h + 3][lid_w + 1].y + 1.h * v[lid_h + 4][lid_w + 1].y;
      break;

    case 3:
      TV0 = -2.h * v[lid_h + 1][lid_w + 0].x - 1.h * v[lid_h + 2][lid_w + 0].x +
            2.h * v[lid_h + 3][lid_w + 0].x + 1.h * v[lid_h + 4][lid_w + 0].x;
      TV1 = -2.h * v[lid_h + 1][lid_w + 0].y - 1.h * v[lid_h + 2][lid_w + 0].y +
            2.h * v[lid_h + 3][lid_w + 0].y + 1.h * v[lid_h + 4][lid_w + 0].y;
      TV2 = -2.h * v[lid_h + 1][lid_w + 0].z - 1.h * v[lid_h + 2][lid_w + 0].z +
            2.h * v[lid_h + 3][lid_w + 0].z + 1.h * v[lid_h + 4][lid_w + 0].z;
      TV3 = -2.h * v[lid_h + 1][lid_w + 0].w - 1.h * v[lid_h + 2][lid_w + 0].w +
            2.h * v[lid_h + 3][lid_w + 0].w + 1.h * v[lid_h + 4][lid_w + 0].w;
      TV4 = -2.h * v[lid_h + 1][lid_w + 1].x - 1.h * v[lid_h + 2][lid_w + 1].x +
            2.h * v[lid_h + 3][lid_w + 1].x + 1.h * v[lid_h + 4][lid_w + 1].x;
      TV5 = -2.h * v[lid_h + 1][lid_w + 1].y - 1.h * v[lid_h + 2][lid_w + 1].y +
            2.h * v[lid_h + 3][lid_w + 1].y + 1.h * v[lid_h + 4][lid_w + 1].y;
      break;

    case 4:
      TV0 = 2.h * v[lid_h + 1][lid_w + 0].x - 1.h * v[lid_h + 2][lid_w + 0].x -
            2.h * v[lid_h + 3][lid_w + 0].x + 1.h * v[lid_h + 4][lid_w + 0].x;
      TV1 = 2.h * v[lid_h + 1][lid_w + 0].y - 1.h * v[lid_h + 2][lid_w + 0].y -
            2.h * v[lid_h + 3][lid_w + 0].y + 1.h * v[lid_h + 4][lid_w + 0].y;
      TV2 = 2.h * v[lid_h + 1][lid_w + 0].z - 1.h * v[lid_h + 2][lid_w + 0].z -
            2.h * v[lid_h + 3][lid_w + 0].z + 1.h * v[lid_h + 4][lid_w + 0].z;
      TV3 = 2.h * v[lid_h + 1][lid_w + 0].w - 1.h * v[lid_h + 2][lid_w + 0].w -
            2.h * v[lid_h + 3][lid_w + 0].w + 1.h * v[lid_h + 4][lid_w + 0].w;
      TV4 = 2.h * v[lid_h + 1][lid_w + 1].x - 1.h * v[lid_h + 2][lid_w + 1].x -
            2.h * v[lid_h + 3][lid_w + 1].x + 1.h * v[lid_h + 4][lid_w + 1].x;
      TV5 = 2.h * v[lid_h + 1][lid_w + 1].y - 1.h * v[lid_h + 2][lid_w + 1].y -
            2.h * v[lid_h + 3][lid_w + 1].y + 1.h * v[lid_h + 4][lid_w + 1].y;
      break;

    case 5:
      TV0 = 4.h * v[lid_h + 1][lid_w + 0].x - 5.h * v[lid_h + 3][lid_w + 0].x +
            1.h * v[lid_h + 5][lid_w + 0].x;
      TV1 = 4.h * v[lid_h + 1][lid_w + 0].y - 5.h * v[lid_h + 3][lid_w + 0].y +
            1.h * v[lid_h + 5][lid_w + 0].y;
      TV2 = 4.h * v[lid_h + 1][lid_w + 0].z - 5.h * v[lid_h + 3][lid_w + 0].z +
            1.h * v[lid_h + 5][lid_w + 0].z;
      TV3 = 4.h * v[lid_h + 1][lid_w + 0].w - 5.h * v[lid_h + 3][lid_w + 0].w +
            1.h * v[lid_h + 5][lid_w + 0].w;
      TV4 = 4.h * v[lid_h + 1][lid_w + 1].x - 5.h * v[lid_h + 3][lid_w + 1].x +
            1.h * v[lid_h + 5][lid_w + 1].x;
      TV5 = 4.h * v[lid_h + 1][lid_w + 1].y - 5.h * v[lid_h + 3][lid_w + 1].y +
            1.h * v[lid_h + 5][lid_w + 1].y;
      break;
    }

    V0 = 4.h * TV0 - 5.h * TV2 + 1.h * TV4;
    V1 = -4.h * TV1 - 4.h * TV2 + 1.h * TV3 + 1.h * TV4;
    V2 = 4.h * TV1 - 4.h * TV2 - 1.h * TV3 + 1.h * TV4;
    V3 = -2.h * TV1 - 1.h * TV2 + 2.h * TV3 + 1.h * TV4;
    V4 = 2.h * TV1 - 1.h * TV2 - 2.h * TV3 + 1.h * TV4;
    V5 = 4.h * TV1 - 5.h * TV3 + 1.h * TV5;

    outputs[0] = V0;
    outputs += K;
    outputs[0] = V1;
    outputs += K;
    outputs[0] = V2;
    outputs += K;
    outputs[0] = V3;
    outputs += K;
    outputs[0] = V4;
    outputs += K;
    outputs[0] = V5;
    outputs += K;
  }
}

__kernel void conv_wino3_filter_tile(__global half *inputs,
                                     __global half *outputs, int K, int C) {

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = get_global_id(0);
  int kc = gid;
  int k = kc / (C);
  if (k >= K)
    return;
  int c = kc - k * (C);

  inputs += (k * C + c) * 3 * 3;
  outputs += k * C + c;

  half w00, w01, w02, w10, w11, w12, w20, w21, w22;

  w00 = inputs[0];
  w01 = inputs[1];
  w02 = inputs[2];
  w10 = inputs[3];
  w11 = inputs[4];
  w12 = inputs[5];
  w20 = inputs[6];
  w21 = inputs[7];
  w22 = inputs[8];

  half out00, out01, out02, out03, out04, out05;

  for (int row = 0; row < 6; row++) {
    switch (row) {
    case 0:
      out00 = (w00) / 16.h;
      out01 = (-w00 - w01 - w02) / 24.h;
      out02 = (-w00 + w01 - w02) / 24.h;
      out03 = (w00 + 2.h * w01 + 4.h * w02) / 96.h;
      out04 = (w00 - 2.h * w01 + 4.h * w02) / 96.h;
      out05 = (w02) / 4.h;
      break;

    case 1:
      out00 = (-w00 - w10 - w20) / 24.h;
      out01 = (w00 + w10 + w20 + w01 + w11 + w21 + w02 + w12 + w22) / 36.h;
      out02 = (w00 + w10 + w20 - w01 - w11 - w21 + w02 + w12 + w22) / 36.h;
      out03 = (-w00 - w10 - w20 + 2.h * (-w01 - w11 - w21) +
               4.h * (-w02 - w12 - w22)) /
              144.h;
      out04 = (-w00 - w10 - w20 + 2.h * (w01 + w11 + w21) +
               4.h * (-w02 - w12 - w22)) /
              144.h;
      out05 = (-w02 - w12 - w22) / 6.h;
      break;

    case 2:
      out00 = (-w00 + w10 - w20) / 24.h;
      out01 = (w00 - w10 + w20 + w01 - w11 + w21 + w02 - w12 + w22) / 36.h;
      out02 = (w00 - w10 + w20 - w01 + w11 - w21 + w02 - w12 + w22) / 36.h;
      out03 = (-w00 + w10 - w20 + 2.h * (-w01 + w11 - w21) +
               4.h * (-w02 + w12 - w22)) /
              144.h;
      out04 = (-w00 + w10 - w20 + 2.h * (w01 - w11 + w21) +
               4.h * (-w02 + w12 - w22)) /
              144.h;
      out05 = (-w02 + w12 - w22) / 6.h;
      break;

    case 3:
      out00 = (w00 + 2.h * w10 + 4.h * w20) / 96.h;
      out01 = (-w00 - 2.h * w10 - 4.h * w20 - w01 - 2.h * w11 - 4.h * w21 -
               w02 - 2.h * w12 - 4.h * w22) /
              144.h;
      out02 = (-w00 - 2.h * w10 - 4.h * w20 + w01 + 2.h * w11 + 4.h * w21 -
               w02 - 2.h * w12 - 4.h * w22) /
              144.h;
      out03 =
          ((w00 + 2.h * w10 + 4.h * w20) + 2.h * (w01 + 2.h * w11 + 4.h * w21) +
           4.h * (w02 + 2.h * w12 + 4.h * w22)) /
          576.h;
      out04 = ((w00 + 2.h * w10 + 4.h * w20) +
               2.h * (-w01 - 2.h * w11 - 4.h * w21) +
               4.h * (w02 + 2.h * w12 + 4.h * w22)) /
              576.h;
      out05 = (w02 + 2.h * w12 + 4.h * w22) / 24.h;
      break;

    case 4:
      out00 = (w00 - 2.h * w10 + 4.h * w20) / 96.h;
      out01 = (-w00 + 2.h * w10 - 4.h * w20 - w01 + 2.h * w11 - 4.h * w21 -
               w02 + 2.h * w12 - 4.h * w22) /
              144.h;
      out02 = (-w00 + 2.h * w10 - 4.h * w20 + w01 - 2.h * w11 + 4.h * w21 -
               w02 + 2.h * w12 - 4.h * w22) /
              144.h;
      out03 =
          ((w00 - 2.h * w10 + 4.h * w20) + 2.h * (w01 - 2.h * w11 + 4.h * w21) +
           4.h * (w02 - 2.h * w12 + 4.h * w22)) /
          576.h;
      out04 = ((w00 - 2.h * w10 + 4.h * w20) +
               2.h * (-w01 + 2.h * w11 - 4.h * w21) +
               4.h * (w02 - 2.h * w12 + 4.h * w22)) /
              576.h;
      out05 = (w02 - 2.h * w12 + 4.h * w22) / 24.h;
      break;

    case 5:
      out00 = (w20) / 4.h;
      out01 = (-w20 - w21 - w22) / 6.h;
      out02 = (-w20 + w21 - w22) / 6.h;
      out03 = (w20 + 2.h * w21 + 4.h * w22) / 24.h;
      out04 = (w20 - 2.h * w21 + 4.h * w22) / 24.h;
      out05 = (w22);
      break;
    }

    outputs[0] = out00;
    outputs += K * C;
    outputs[0] = out01;
    outputs += K * C;
    outputs[0] = out02;
    outputs += K * C;
    outputs[0] = out03;
    outputs += K * C;
    outputs[0] = out04;
    outputs += K * C;
    outputs[0] = out05;
    outputs += K * C;
  }
}

__kernel void conv_wino3_gemm_opt(__global half *X_A, // X instance of MxK mat
                                  __global half *X_B, // X instance of KxN mat
                                  __global half *X_C, int X, int M, int N,
                                  int K) // X instance of MxN mat
{
  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k, idx;

  // How many output pixels per therad?
  // PT_M x PT_N
  float2 reg_C[PT_M][PT_N_OPT];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N_OPT; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N_OPT;
  int mid = lid / TP_N_OPT;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local float local_A[TILE_M * TILE_K];
  __local float local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *A = X_A + xid * M * K;
  __global half *B = X_B + xid * K * N;
  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {
#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] = SAFE_LOAD_HALF(
              A, M, K, (toff_m + m + warp_id), (toff_k + k + wid));
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          (local_B)[((k + warp_id) * TILE_N + n + wid)] = SAFE_LOAD_HALF(
              B, K, N, (toff_k + k + warp_id), (toff_n + n + wid));
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
        float2 reg = local_A[(m * TP_M + mid) * TILE_K + k];
#pragma unroll
        for (n = 0; n < PT_N_OPT; n++) {
          idx = k * TILE_N + n * TP_N_OPT * 2 + nid * 2;
          float2 regV = vload2(idx >> 1, local_B);
          reg_C[m][n] += reg * regV;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N_OPT; n++) {
      SAFE_VEC2_STORE_HALF(C, M, N, (toff_m + m * TP_M + mid),
                           (toff_n + n * TP_N_OPT * 2 + nid * 2), reg_C[m][n]);
    }
  }
}

__kernel void
conv_wino3_gemm(__global half *X_A, // X instance of MxK mat
                        __global half *X_B, // X instance of KxN mat
                        __global half *X_C, int X, int M, int N,
                        int K) // X instance of MxN mat
{
  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k;

  // How many output pixels per therad?
  // PT_M x PT_N
  float reg_C[PT_M][PT_N];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local float local_A[TILE_M * TILE_K];
  __local float local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *A = X_A + xid * M * K;
  __global half *B = X_B + xid * K * N;
  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {
#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] = SAFE_LOAD_HALF(
              A, M, K, (toff_m + m + warp_id), (toff_k + k + wid));
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          (local_B)[((k + warp_id) * TILE_N + n + wid)] = SAFE_LOAD_HALF(
              B, K, N, (toff_k + k + warp_id), (toff_n + n + wid));
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
        float reg = local_A[(m * TP_M + mid) * TILE_K + k];
#pragma unroll
        for (n = 0; n < PT_N; n++) {
          reg_C[m][n] += reg * local_B[k * TILE_N + n * TP_N + nid];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE_HALF(C, M, N, (toff_m + m * TP_M + mid),
                      (toff_n + n * TP_N + nid), reg_C[m][n]);

    }
  }
}

__kernel void conv_wino3_data_untile_no_bias(__global half *inputs,
                                             __global half *outputs, int K,
                                             int H, int W) {

  int TP = (H + 1) / 4;
  int TQ = (W + 1) / 4;

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = bid * lsz + lid;
  int ktptq = gid;
  int k = ktptq / (TP * TQ);
  if (k >= K)
    return;
  int tptq = ktptq - k * (TP * TQ);
  int tp = tptq / (TQ);
  int tq = tptq - tp * (TQ);
  int p = tp * 4, q = tq * 4;

  int P = H - 2;
  int Q = W - 2;
  inputs += (k * TP + tp) * TQ + tq;
  outputs += (k * P + p) * Q + q;

  half4 V0 = (half4)(0, 0, 0, 0);
  half4 V1 = (half4)(0, 0, 0, 0);
  half4 V2 = (half4)(0, 0, 0, 0);
  half4 V3 = (half4)(0, 0, 0, 0);
  half m0, m1, m2, m3, m4, m5;

  for (int i = 0; i < 6; ++i) {

    m0 = inputs[0];
    inputs += K * TP * TQ;
    m1 = inputs[0];
    inputs += K * TP * TQ;
    m2 = inputs[0];
    inputs += K * TP * TQ;
    m3 = inputs[0];
    inputs += K * TP * TQ;
    m4 = inputs[0];
    inputs += K * TP * TQ;
    m5 = inputs[0];
    inputs += K * TP * TQ;

    switch (i) {
    case 0:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2 * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4 * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8 * m4 + m5;
      break;
    case 1:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += m0 + m1 + m2 + m3 + m4;
      V1.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V1.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V1.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V2.x += m0 + m1 + m2 + m3 + m4;
      V2.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V2.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V2.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V3.x += m0 + m1 + m2 + m3 + m4;
      V3.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V3.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V3.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;
      break;
    case 2:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += -m0 - m1 - m2 - m3 - m4;
      V1.y += -m1 + m2 - 2.h * m3 + 2.h * m4;
      V1.z += -m1 - m2 - 4.h * m3 - 4.h * m4;
      V1.w += -m1 + m2 - 8.h * m3 + 8.h * m4 - m5;

      V2.x += m0 + m1 + m2 + m3 + m4;
      V2.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V2.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V2.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V3.x += -m0 - m1 - m2 - m3 - m4;
      V3.y += -m1 + m2 - 2.h * m3 + 2.h * m4;
      V3.z += -m1 - m2 - 4.h * m3 - 4.h * m4;
      V3.w += -m1 + m2 - 8.h * m3 + 8.h * m4 - m5;
      break;
    case 3:

      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += 2.h * m0 + 2.h * m1 + 2.h * m2 + 2.h * m3 + 2.h * m4;
      V1.y += 2.h * m1 - 2.h * m2 + 4.h * m3 - 4.h * m4;
      V1.z += 2.h * m1 + 2.h * m2 + 8.h * m3 + 8.h * m4;
      V1.w += 2.h * m1 - 2.h * m2 + 16.h * m3 - 16.h * m4 + 2.h * m5;

      V2.x += 4.h * m0 + 4.h * m1 + 4.h * m2 + 4.h * m3 + 4.h * m4;
      V2.y += 4.h * m1 - 4.h * m2 + 8.h * m3 - 8.h * m4;
      V2.z += 4.h * m1 + 4.h * m2 + 16.h * m3 + 16.h * m4;
      V2.w += 4.h * m1 - 4.h * m2 + 32.h * m3 - 32.h * m4 + 4.h * m5;

      V3.x += 8.h * m0 + 8.h * m1 + 8.h * m2 + 8.h * m3 + 8.h * m4;
      V3.y += 8.h * m1 - 8.h * m2 + 16.h * m3 - 16.h * m4;
      V3.z += 8.h * m1 + 8.h * m2 + 32.h * m3 + 32.h * m4;
      V3.w += 8.h * m1 - 8.h * m2 + 64.h * m3 - 64.h * m4 + 8.h * m5;
      break;
    case 4:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += -2.h * m0 - 2.h * m1 - 2.h * m2 - 2.h * m3 - 2.h * m4;
      V1.y += -2.h * m1 + 2.h * m2 - 4.h * m3 + 4.h * m4;
      V1.z += -2.h * m1 - 2.h * m2 - 8.h * m3 - 8.h * m4;
      V1.w += -2.h * m1 + 2.h * m2 - 16.h * m3 + 16.h * m4 - 2.h * m5;

      V2.x += 4.h * m0 + 4.h * m1 + 4.h * m2 + 4.h * m3 + 4.h * m4;
      V2.y += 4.h * m1 - 4.h * m2 + 8.h * m3 - 8.h * m4;
      V2.z += 4.h * m1 + 4.h * m2 + 16.h * m3 + 16.h * m4;
      V2.w += 4.h * m1 - 4.h * m2 + 32.h * m3 - 32.h * m4 + 4.h * m5;

      V3.x += -8.h * m0 - 8.h * m1 - 8.h * m2 - 8.h * m3 - 8.h * m4;
      V3.y += -8.h * m1 + 8.h * m2 - 16.h * m3 + 16.h * m4;
      V3.z += -8.h * m1 - 8.h * m2 - 32.h * m3 - 32.h * m4;
      V3.w += -8.h * m1 + 8.h * m2 - 64.h * m3 + 64.h * m4 - 8.h * m5;
      break;
    case 5:
      V3.x += m0 + m1 + m2 + m3 + m4;
      V3.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V3.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V3.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;
      break;
    }
  }

  if (p + 3 < P && q + 3 < Q) {
    half *ptr = outputs;
    vstore4(V0, 0, ptr);
    ptr += Q;
    vstore4(V1, 0, ptr);
    ptr += Q;
    vstore4(V2, 0, ptr);
    ptr += Q;
    vstore4(V3, 0, ptr);

  } else {
    if (p + 0 < P && q + 0 < Q)
      outputs[0 * Q + 0] = V0.x;
    if (p + 0 < P && q + 1 < Q)
      outputs[0 * Q + 1] = V0.y;
    if (p + 0 < P && q + 2 < Q)
      outputs[0 * Q + 2] = V0.z;
    if (p + 0 < P && q + 3 < Q)
      outputs[0 * Q + 3] = V0.w;

    if (p + 1 < P && q + 0 < Q)
      outputs[1 * Q + 0] = V1.x;
    if (p + 1 < P && q + 1 < Q)
      outputs[1 * Q + 1] = V1.y;
    if (p + 1 < P && q + 2 < Q)
      outputs[1 * Q + 2] = V1.z;
    if (p + 1 < P && q + 3 < Q)
      outputs[1 * Q + 3] = V1.w;

    if (p + 2 < P && q + 0 < Q)
      outputs[2 * Q + 0] = V2.x;
    if (p + 2 < P && q + 1 < Q)
      outputs[2 * Q + 1] = V2.y;
    if (p + 2 < P && q + 2 < Q)
      outputs[2 * Q + 2] = V2.z;
    if (p + 2 < P && q + 3 < Q)
      outputs[2 * Q + 3] = V2.w;

    if (p + 3 < P && q + 0 < Q)
      outputs[3 * Q + 0] = V3.x;
    if (p + 3 < P && q + 1 < Q)
      outputs[3 * Q + 1] = V3.y;
    if (p + 3 < P && q + 2 < Q)
      outputs[3 * Q + 2] = V3.z;
    if (p + 3 < P && q + 3 < Q)
      outputs[3 * Q + 3] = V3.w;
  }
}

__kernel void conv_wino3_data_untile(__global half *inputs,
                                     __global half *outputs,
                                     __global half *bias, int K, int H, int W) {

  int TP = (H + 1) / 4;
  int TQ = (W + 1) / 4;

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = bid * lsz + lid;
  int ktptq = gid;
  int k = ktptq / (TP * TQ);
  if (k >= K)
    return;
  int tptq = ktptq - k * (TP * TQ);
  int tp = tptq / (TQ);
  int tq = tptq - tp * (TQ);
  int p = tp * 4, q = tq * 4;

  int P = (H - 3) + 1;
  int Q = (W - 3) + 1;
  inputs += (k * TP + tp) * TQ + tq;
  outputs += (k * P + p) * Q + q;

  half4 V0 = (half4)(0, 0, 0, 0);
  half4 V1 = (half4)(0, 0, 0, 0);
  half4 V2 = (half4)(0, 0, 0, 0);
  half4 V3 = (half4)(0, 0, 0, 0);
  half m0, m1, m2, m3, m4, m5;

  for (int i = 0; i < 6; ++i) {

    m0 = inputs[0];
    inputs += K * TP * TQ;
    m1 = inputs[0];
    inputs += K * TP * TQ;
    m2 = inputs[0];
    inputs += K * TP * TQ;
    m3 = inputs[0];
    inputs += K * TP * TQ;
    m4 = inputs[0];
    inputs += K * TP * TQ;
    m5 = inputs[0];
    inputs += K * TP * TQ;

    switch (i) {
    case 0:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2 * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4 * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8 * m4 + m5;
      break;
    case 1:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += m0 + m1 + m2 + m3 + m4;
      V1.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V1.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V1.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V2.x += m0 + m1 + m2 + m3 + m4;
      V2.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V2.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V2.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V3.x += m0 + m1 + m2 + m3 + m4;
      V3.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V3.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V3.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;
      break;
    case 2:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += -m0 - m1 - m2 - m3 - m4;
      V1.y += -m1 + m2 - 2.h * m3 + 2.h * m4;
      V1.z += -m1 - m2 - 4.h * m3 - 4.h * m4;
      V1.w += -m1 + m2 - 8.h * m3 + 8.h * m4 - m5;

      V2.x += m0 + m1 + m2 + m3 + m4;
      V2.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V2.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V2.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V3.x += -m0 - m1 - m2 - m3 - m4;
      V3.y += -m1 + m2 - 2.h * m3 + 2.h * m4;
      V3.z += -m1 - m2 - 4.h * m3 - 4.h * m4;
      V3.w += -m1 + m2 - 8.h * m3 + 8.h * m4 - m5;
      break;
    case 3:

      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += 2.h * m0 + 2.h * m1 + 2.h * m2 + 2.h * m3 + 2.h * m4;
      V1.y += 2.h * m1 - 2.h * m2 + 4.h * m3 - 4.h * m4;
      V1.z += 2.h * m1 + 2.h * m2 + 8.h * m3 + 8.h * m4;
      V1.w += 2.h * m1 - 2.h * m2 + 16.h * m3 - 16.h * m4 + 2.h * m5;

      V2.x += 4.h * m0 + 4.h * m1 + 4.h * m2 + 4.h * m3 + 4.h * m4;
      V2.y += 4.h * m1 - 4.h * m2 + 8.h * m3 - 8.h * m4;
      V2.z += 4.h * m1 + 4.h * m2 + 16.h * m3 + 16.h * m4;
      V2.w += 4.h * m1 - 4.h * m2 + 32.h * m3 - 32.h * m4 + 4.h * m5;

      V3.x += 8.h * m0 + 8.h * m1 + 8.h * m2 + 8.h * m3 + 8.h * m4;
      V3.y += 8.h * m1 - 8.h * m2 + 16.h * m3 - 16.h * m4;
      V3.z += 8.h * m1 + 8.h * m2 + 32.h * m3 + 32.h * m4;
      V3.w += 8.h * m1 - 8.h * m2 + 64.h * m3 - 64.h * m4 + 8.h * m5;
      break;
    case 4:
      V0.x += m0 + m1 + m2 + m3 + m4;
      V0.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V0.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V0.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;

      V1.x += -2.h * m0 - 2.h * m1 - 2.h * m2 - 2.h * m3 - 2.h * m4;
      V1.y += -2.h * m1 + 2.h * m2 - 4.h * m3 + 4.h * m4;
      V1.z += -2.h * m1 - 2.h * m2 - 8.h * m3 - 8.h * m4;
      V1.w += -2.h * m1 + 2.h * m2 - 16.h * m3 + 16.h * m4 - 2.h * m5;

      V2.x += 4.h * m0 + 4.h * m1 + 4.h * m2 + 4.h * m3 + 4.h * m4;
      V2.y += 4.h * m1 - 4.h * m2 + 8.h * m3 - 8.h * m4;
      V2.z += 4.h * m1 + 4.h * m2 + 16.h * m3 + 16.h * m4;
      V2.w += 4.h * m1 - 4.h * m2 + 32.h * m3 - 32.h * m4 + 4.h * m5;

      V3.x += -8.h * m0 - 8.h * m1 - 8.h * m2 - 8.h * m3 - 8.h * m4;
      V3.y += -8.h * m1 + 8.h * m2 - 16.h * m3 + 16.h * m4;
      V3.z += -8.h * m1 - 8.h * m2 - 32.h * m3 - 32.h * m4;
      V3.w += -8.h * m1 + 8.h * m2 - 64.h * m3 + 64.h * m4 - 8.h * m5;
      break;
    case 5:
      V3.x += m0 + m1 + m2 + m3 + m4;
      V3.y += m1 - m2 + 2.h * m3 - 2.h * m4;
      V3.z += m1 + m2 + 4.h * m3 + 4.h * m4;
      V3.w += m1 - m2 + 8.h * m3 - 8.h * m4 + m5;
      break;
    }
  }

  half b = bias[k];
  if (p + 3 < P && q + 3 < Q) {
    half4 B = (half4)(b, b, b, b);
    V0 += B;
    V1 += B;
    V2 += B;
    V3 += B;

    half *ptr = outputs;
    vstore4(V0, 0, ptr);
    ptr += Q;
    vstore4(V1, 0, ptr);
    ptr += Q;
    vstore4(V2, 0, ptr);
    ptr += Q;
    vstore4(V3, 0, ptr);

  } else {
    if (p + 0 < P && q + 0 < Q)
      outputs[0 * Q + 0] = V0.x + b;
    if (p + 0 < P && q + 1 < Q)
      outputs[0 * Q + 1] = V0.y + b;
    if (p + 0 < P && q + 2 < Q)
      outputs[0 * Q + 2] = V0.z + b;
    if (p + 0 < P && q + 3 < Q)
      outputs[0 * Q + 3] = V0.w + b;

    if (p + 1 < P && q + 0 < Q)
      outputs[1 * Q + 0] = V1.x + b;
    if (p + 1 < P && q + 1 < Q)
      outputs[1 * Q + 1] = V1.y + b;
    if (p + 1 < P && q + 2 < Q)
      outputs[1 * Q + 2] = V1.z + b;
    if (p + 1 < P && q + 3 < Q)
      outputs[1 * Q + 3] = V1.w + b;

    if (p + 2 < P && q + 0 < Q)
      outputs[2 * Q + 0] = V2.x + b;
    if (p + 2 < P && q + 1 < Q)
      outputs[2 * Q + 1] = V2.y + b;
    if (p + 2 < P && q + 2 < Q)
      outputs[2 * Q + 2] = V2.z + b;
    if (p + 2 < P && q + 3 < Q)
      outputs[2 * Q + 3] = V2.w + b;

    if (p + 3 < P && q + 0 < Q)
      outputs[3 * Q + 0] = V3.x + b;
    if (p + 3 < P && q + 1 < Q)
      outputs[3 * Q + 1] = V3.y + b;
    if (p + 3 < P && q + 2 < Q)
      outputs[3 * Q + 2] = V3.z + b;
    if (p + 3 < P && q + 3 < Q)
      outputs[3 * Q + 3] = V3.w + b;
  }
}

__kernel void conv_implicit_gemm_img(__read_only image1d_buffer_t input,
                                     __global half *output,
                                     __read_only image1d_buffer_t filter,
                                     int CIN, int COUT, int IH, int IW, int OH,
                                     int OW, int R) {
  int M = COUT;
  int N = OH * OW;
  int K = CIN * R * R;
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0));
  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  half2 reg_C[PT_M][PT_N_OPT];
#pragma unroll
  for (unsigned int i = 0; i < PT_M; i++)
    for (unsigned int j = 0; j < PT_N_OPT; j++)
      reg_C[i][j] = 0;

  int nid = lid % TP_N_OPT;
  int mid = lid / TP_N_OPT;
  int nbid = bid % NB_P_INST;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];

  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;

  // x(0~inputchannel * filterheight * filterwidth), y(width * height)
  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
#pragma unroll
    for (unsigned int m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
      for (unsigned k = 0; k < TILE_K; k += WARP_SIZE) {
        int idx = (toff_m + m + warp_id) * K + (toff_k + k + wid);
        half2 regA = read_imageh(filter, idx >> 1).xy;
        local_A[(m + warp_id) * TILE_K + k + wid] =
            (idx % 2 == 0) ? regA.x : regA.y;
      }
    }

#pragma unroll
    for (unsigned k = 0; k < TILE_K; k += WARP_CNT) {
      int kid = toff_k + k + warp_id; // 0 ~ channelin * fw * fh;
      int c = kid / (R * R);
      int fh = (kid / R) % R;
      int fw = kid % R;

#pragma unroll
      for (unsigned n = 0; n < TILE_N; n += WARP_SIZE) {
        int nid = toff_n + n + wid; // 0 ~ OH * OW
        int oh = nid / OW;
        int ow = nid % OW;
        int ih = oh + fh;
        int iw = ow + fw;
        int idx = c * IH * IW + ih * IW + iw;

        half2 regB = read_imageh(input, idx >> 1).xy;
        local_B[((k + warp_id) * TILE_N + n + wid)] =
            (idx % 2 == 0) ? regB.x : regB.y;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // matrix-multiplication using local memory
#pragma unroll
    for (unsigned int k = 0; k < TILE_K; k++) {
#pragma unroll
      for (unsigned int m = 0; m < PT_M; m++) {
        half2 reg = local_A[(m * TP_M + mid) * TILE_K + k];
#pragma unroll
        for (unsigned int n = 0; n < PT_N_OPT; n++) {
          int idx = k * TILE_N + n * TP_N_OPT * 2 + nid * 2;
          half2 regV = vload2(idx >> 1, local_B);
          reg_C[m][n] += reg * regV;
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (unsigned int m = 0; m < PT_M; m++) {
#pragma unroll
    for (unsigned int n = 0; n < PT_N_OPT; n++) {
      SAFE_VEC2_STORE(output, M, N, (toff_m + m * TP_M + mid),
                      (toff_n + n * TP_N_OPT * 2 + nid * 2), reg_C[m][n]);
    }
  }
}

__kernel void conv_implicit_gemm(__global half *input, __global half *output,
                                 __global half *filter, int CIN, int COUT,
                                 int IH, int IW, int OH, int OW, int R) {
  __global half *X_A = filter;
  __global half *X_B = input;
  __global half *X_C = output;
  int M = COUT;
  int N = OH * OW;
  int K = CIN * R * R;
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ
  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;
  int m, n, k;

  // How many output pixels per therad?
  // PT_M x PT_N
  half reg_C[PT_M][PT_N];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *A = X_A;
  __global half *B = X_B;
  __global half *C = X_C;

  // x(0~inputchannel * filterheight * filterwidth), y(width * height)

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {
#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              SAFE_LOAD(A, M, K, (toff_m + m + warp_id), (toff_k + k + wid));
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {

          int kid = toff_k + k + warp_id; // 0 ~ channelin * fw * fh;
          int nid = toff_n + n + wid;     // 0 ~ OH * OW
          int oh = nid / OW;
          int ow = nid % OW;
          int c = kid / (R * R);
          int fh = (kid / R) % R;
          int fw = kid % R;
          int ih = oh + fh;
          int iw = ow + fw;
          int valid =
              (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW) && (c < CIN);
          int idx = c * IH * IW + ih * IW + iw;

          (local_B)[((k + warp_id) * TILE_N + n + wid)] = valid ? B[idx] : 0.h;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication --> #pragma 효과있음
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
        half reg = local_A[(m * TP_M + mid) * TILE_K + k];
#pragma unroll
        for (n = 0; n < PT_N; n++) {
          reg_C[m][n] += reg * local_B[k * TILE_N + n * TP_N + nid];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid),
                 reg_C[m][n]);
    }
  }
}

__kernel void conv_wino5_filter_tile(__global half *inputs,
                                     __global half *outputs, int K, int C) {
  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = get_global_id(0);
  int kc = gid;
  int k = kc / (C);
  if (k >= K)
    return;
  int c = kc - k * (C);

  inputs += (k * C + c) * 5 * 5;
  outputs += k * C + c;

  half4 w00, w10, w20, w30, w40;
  half w01, w11, w21, w31, w41;

  w00 = vload4(0, inputs);
  w01 = *(inputs + 4);
  inputs += 5;
  w10 = vload4(0, inputs);
  w11 = *(inputs + 4);
  inputs += 5;
  w20 = vload4(0, inputs);
  w21 = *(inputs + 4);
  inputs += 5;
  w30 = vload4(0, inputs);
  w31 = *(inputs + 4);
  inputs += 5;
  w40 = vload4(0, inputs);
  w41 = *(inputs + 4);

  half8 out0;

  for (int row = 0; row < 8; row++) {
    switch (row) {
    case 0:
      // Row 0
      out0.s0 = w00.s0;
      out0.s1 = -2.h * (w00.s0 + w00.s1 + w00.s2 + w00.s3 + w01) / 9.h;
      out0.s2 = -2.h * (w00.s0 - w00.s1 + w00.s2 - w00.s3 + w01) / 9.h;
      out0.s3 =
          (w00.s0 + 2.h * w00.s1 + 4.h * w00.s2 + 8.h * w00.s3 + 16.h * w01) /
          90.h;
      out0.s4 =
          (w00.s0 - 2.h * w00.s1 + 4.h * w00.s2 - 8.h * w00.s3 + 16.h * w01) /
          90.h;
      out0.s5 =
          (16.h * w00.s0 + 8.h * w00.s1 + 4.h * w00.s2 + 2.h * w00.s3 + w01) /
          180.h;
      out0.s6 =
          (16.h * w00.s0 - 8.h * w00.s1 + 4.h * w00.s2 - 2.h * w00.s3 + w01) /
          180.h;
      out0.s7 = w01;
      break;

    case 1:
      out0.s0 = -2.h * (w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) / 9.h;
      out0.s1 = 4.h *
                ((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) +
                 (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) +
                 (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) +
                 (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) +
                 (w01 + w11 + w21 + w31 + w41)) /
                81.h;
      out0.s2 = 4.h *
                ((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) -
                 (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) +
                 (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) -
                 (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) +
                 (w01 + w11 + w21 + w31 + w41)) /
                81.h;
      out0.s3 = -((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) +
                  2.h * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) +
                  4.h * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) +
                  8.h * (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) +
                  16.h * (w01 + w11 + w21 + w31 + w41)) /
                405.h;
      out0.s4 = -((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) -
                  2.h * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) +
                  4.h * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) -
                  8.h * (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) +
                  16.h * (w01 + w11 + w21 + w31 + w41)) /
                405.h;
      out0.s5 = -(16.h * (w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) +
                  8.h * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) +
                  4.h * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) +
                  2.h * (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) +
                  (w01 + w11 + w21 + w31 + w41)) /
                810.h;
      out0.s6 = -(16.h * (w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) -
                  8.h * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) +
                  4.h * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) -
                  2.h * (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) +
                  (w01 + w11 + w21 + w31 + w41)) /
                810.h;
      out0.s7 = -2.h * (w01 + w11 + w21 + w31 + w41) / 9.h;
      break;

    case 2:
      // Row 2
      out0.s0 = -2.h * (w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) / 9.h;
      out0.s1 = 4.h *
                ((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) +
                 (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) +
                 (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) +
                 (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) +
                 (w01 - w11 + w21 - w31 + w41)) /
                81.h;
      out0.s2 = 4.h *
                ((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) -
                 (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) +
                 (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) -
                 (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) +
                 (w01 - w11 + w21 - w31 + w41)) /
                81.h;
      out0.s3 = -((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) +
                  2.h * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) +
                  4.h * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) +
                  8.h * (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) +
                  16.h * (w01 - w11 + w21 - w31 + w41)) /
                405.h;
      out0.s4 = -((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) -
                  2.h * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) +
                  4.h * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) -
                  8.h * (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) +
                  16.h * (w01 - w11 + w21 - w31 + w41)) /
                405.h;
      out0.s5 = -(16.h * (w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) +
                  8.h * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) +
                  4.h * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) +
                  2.h * (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) +
                  (w01 - w11 + w21 - w31 + w41)) /
                810.h;
      out0.s6 = -(16.h * (w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) -
                  8.h * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) +
                  4.h * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) -
                  2.h * (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) +
                  (w01 - w11 + w21 - w31 + w41)) /
                810.h;
      out0.s7 = -2.h * (w01 - w11 + w21 - w31 + w41) / 9.h;
      break;

    case 3:
      // Row 3
      out0.s0 = (w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
                 16.h * w40.s0) /
                90.h;
      out0.s1 = -((w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
                   16.h * w40.s0) +
                  (w00.s1 + 2.h * w10.s1 + 4.h * w20.s1 + 8.h * w30.s1 +
                   16.h * w40.s1) +
                  (w00.s2 + 2.h * w10.s2 + 4.h * w20.s2 + 8.h * w30.s2 +
                   16.h * w40.s2) +
                  (w00.s3 + 2.h * w10.s3 + 4.h * w20.s3 + 8.h * w30.s3 +
                   16.h * w40.s3) +
                  (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41)) /
                405.h;
      out0.s2 = -((w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
                   16.h * w40.s0) -
                  (w00.s1 + 2.h * w10.s1 + 4.h * w20.s1 + 8.h * w30.s1 +
                   16.h * w40.s1) +
                  (w00.s2 + 2.h * w10.s2 + 4.h * w20.s2 + 8.h * w30.s2 +
                   16.h * w40.s2) -
                  (w00.s3 + 2.h * w10.s3 + 4.h * w20.s3 + 8.h * w30.s3 +
                   16.h * w40.s3) +
                  (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41)) /
                405.h;
      out0.s3 =
          ((w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
            16.h * w40.s0) +
           2.h * (w00.s1 + 2.h * w10.s1 + 4.h * w20.s1 + 8.h * w30.s1 +
                  16.h * w40.s1) +
           4.h * (w00.s2 + 2.h * w10.s2 + 4.h * w20.s2 + 8.h * w30.s2 +
                  16.h * w40.s2) +
           8.h * (w00.s3 + 2.h * w10.s3 + 4.h * w20.s3 + 8.h * w30.s3 +
                  16.h * w40.s3) +
           16.h * (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41)) /
          8100.h;
      out0.s4 =
          ((w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
            16.h * w40.s0) -
           2.h * (w00.s1 + 2.h * w10.s1 + 4.h * w20.s1 + 8.h * w30.s1 +
                  16.h * w40.s1) +
           4.h * (w00.s2 + 2.h * w10.s2 + 4.h * w20.s2 + 8.h * w30.s2 +
                  16.h * w40.s2) -
           8.h * (w00.s3 + 2.h * w10.s3 + 4.h * w20.s3 + 8.h * w30.s3 +
                  16.h * w40.s3) +
           16.h * (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41)) /
          8100.h;
      out0.s5 = (16.h * (w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
                         16.h * w40.s0) +
                 8.h * (w00.s1 + 2.h * w10.s1 + 4.h * w20.s1 + 8.h * w30.s1 +
                        16.h * w40.s1) +
                 4.h * (w00.s2 + 2.h * w10.s2 + 4.h * w20.s2 + 8.h * w30.s2 +
                        16.h * w40.s2) +
                 2.h * (w00.s3 + 2.h * w10.s3 + 4.h * w20.s3 + 8.h * w30.s3 +
                        16.h * w40.s3) +
                 (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41)) /
                16200.h;
      out0.s6 = (16.h * (w00.s0 + 2.h * w10.s0 + 4.h * w20.s0 + 8.h * w30.s0 +
                         16.h * w40.s0) -
                 8.h * (w00.s1 + 2.h * w10.s1 + 4.h * w20.s1 + 8.h * w30.s1 +
                        16.h * w40.s1) +
                 4.h * (w00.s2 + 2.h * w10.s2 + 4.h * w20.s2 + 8.h * w30.s2 +
                        16.h * w40.s2) -
                 2.h * (w00.s3 + 2.h * w10.s3 + 4.h * w20.s3 + 8.h * w30.s3 +
                        16.h * w40.s3) +
                 (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41)) /
                16200.h;
      out0.s7 = (w01 + 2.h * w11 + 4.h * w21 + 8.h * w31 + 16.h * w41) / 90.h;
      break;

    case 4:
      // Row 4
      out0.s0 = (w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
                 16.h * w40.s0) /
                90.h;
      out0.s1 = -((w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
                   16.h * w40.s0) +
                  (w00.s1 - 2.h * w10.s1 + 4.h * w20.s1 - 8.h * w30.s1 +
                   16.h * w40.s1) +
                  (w00.s2 - 2.h * w10.s2 + 4.h * w20.s2 - 8.h * w30.s2 +
                   16.h * w40.s2) +
                  (w00.s3 - 2.h * w10.s3 + 4.h * w20.s3 - 8.h * w30.s3 +
                   16.h * w40.s3) +
                  (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41)) /
                405.h;
      out0.s2 = -((w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
                   16.h * w40.s0) -
                  (w00.s1 - 2.h * w10.s1 + 4.h * w20.s1 - 8.h * w30.s1 +
                   16.h * w40.s1) +
                  (w00.s2 - 2.h * w10.s2 + 4.h * w20.s2 - 8.h * w30.s2 +
                   16.h * w40.s2) -
                  (w00.s3 - 2.h * w10.s3 + 4.h * w20.s3 - 8.h * w30.s3 +
                   16.h * w40.s3) +
                  (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41)) /
                405.h;
      out0.s3 =
          ((w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
            16.h * w40.s0) +
           2.h * (w00.s1 - 2.h * w10.s1 + 4.h * w20.s1 - 8.h * w30.s1 +
                  16.h * w40.s1) +
           4.h * (w00.s2 - 2.h * w10.s2 + 4.h * w20.s2 - 8.h * w30.s2 +
                  16.h * w40.s2) +
           8.h * (w00.s3 - 2.h * w10.s3 + 4.h * w20.s3 - 8.h * w30.s3 +
                  16.h * w40.s3) +
           16.h * (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41)) /
          8100.h;
      out0.s4 =
          ((w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
            16.h * w40.s0) -
           2.h * (w00.s1 - 2.h * w10.s1 + 4.h * w20.s1 - 8.h * w30.s1 +
                  16.h * w40.s1) +
           4.h * (w00.s2 - 2.h * w10.s2 + 4.h * w20.s2 - 8.h * w30.s2 +
                  16.h * w40.s2) -
           8.h * (w00.s3 - 2.h * w10.s3 + 4.h * w20.s3 - 8.h * w30.s3 +
                  16.h * w40.s3) +
           16.h * (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41)) /
          8100.h;
      out0.s5 = (16.h * (w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
                         16.h * w40.s0) +
                 8.h * (w00.s1 - 2.h * w10.s1 + 4.h * w20.s1 - 8.h * w30.s1 +
                        16.h * w40.s1) +
                 4.h * (w00.s2 - 2.h * w10.s2 + 4.h * w20.s2 - 8.h * w30.s2 +
                        16.h * w40.s2) +
                 2.h * (w00.s3 - 2.h * w10.s3 + 4.h * w20.s3 - 8.h * w30.s3 +
                        16.h * w40.s3) +
                 (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41)) /
                16200.h;
      out0.s6 = (16.h * (w00.s0 - 2.h * w10.s0 + 4.h * w20.s0 - 8.h * w30.s0 +
                         16.h * w40.s0) -
                 8.h * (w00.s1 - 2.h * w10.s1 + 4.h * w20.s1 - 8.h * w30.s1 +
                        16.h * w40.s1) +
                 4.h * (w00.s2 - 2.h * w10.s2 + 4.h * w20.s2 - 8.h * w30.s2 +
                        16.h * w40.s2) -
                 2.h * (w00.s3 - 2.h * w10.s3 + 4.h * w20.s3 - 8.h * w30.s3 +
                        16.h * w40.s3) +
                 (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41)) /
                16200.h;
      out0.s7 = (w01 - 2.h * w11 + 4.h * w21 - 8.h * w31 + 16.h * w41) / 90.h;
      break;

    case 5:
      // Row 5
      out0.s0 = (16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 + 2.h * w30.s0 +
                 w40.s0) /
                180.h;
      out0.s1 = -((16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 + 2.h * w30.s0 +
                   w40.s0) +
                  (16.h * w00.s1 + 8.h * w10.s1 + 4.h * w20.s1 + 2.h * w30.s1 +
                   w40.s1) +
                  (16.h * w00.s2 + 8.h * w10.s2 + 4.h * w20.s2 + 2.h * w30.s2 +
                   w40.s2) +
                  (16.h * w00.s3 + 8.h * w10.s3 + 4.h * w20.s3 + 2.h * w30.s3 +
                   w40.s3) +
                  (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41)) /
                810.h;
      out0.s2 = -((16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 + 2.h * w30.s0 +
                   w40.s0) -
                  (16.h * w00.s1 + 8.h * w10.s1 + 4.h * w20.s1 + 2.h * w30.s1 +
                   w40.s1) +
                  (16.h * w00.s2 + 8.h * w10.s2 + 4.h * w20.s2 + 2.h * w30.s2 +
                   w40.s2) -
                  (16.h * w00.s3 + 8.h * w10.s3 + 4.h * w20.s3 + 2.h * w30.s3 +
                   w40.s3) +
                  (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41)) /
                810.h;
      out0.s3 =
          ((16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 + 2.h * w30.s0 +
            w40.s0) +
           2.h * (16.h * w00.s1 + 8.h * w10.s1 + 4.h * w20.s1 + 2.h * w30.s1 +
                  w40.s1) +
           4.h * (16.h * w00.s2 + 8.h * w10.s2 + 4.h * w20.s2 + 2.h * w30.s2 +
                  w40.s2) +
           8.h * (16.h * w00.s3 + 8.h * w10.s3 + 4.h * w20.s3 + 2.h * w30.s3 +
                  w40.s3) +
           16.h * (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41)) /
          16200.h;
      out0.s4 =
          ((16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 + 2.h * w30.s0 +
            w40.s0) -
           2.h * (16.h * w00.s1 + 8.h * w10.s1 + 4.h * w20.s1 + 2.h * w30.s1 +
                  w40.s1) +
           4.h * (16.h * w00.s2 + 8.h * w10.s2 + 4.h * w20.s2 + 2.h * w30.s2 +
                  w40.s2) -
           8.h * (16.h * w00.s3 + 8.h * w10.s3 + 4.h * w20.s3 + 2.h * w30.s3 +
                  w40.s3) +
           16.h * (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41)) /
          16200.h;
      out0.s5 = (16.h * (16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 +
                         2.h * w30.s0 + w40.s0) +
                 8.h * (16.h * w00.s1 + 8.h * w10.s1 + 4.h * w20.s1 +
                        2.h * w30.s1 + w40.s1) +
                 4.h * (16.h * w00.s2 + 8.h * w10.s2 + 4.h * w20.s2 +
                        2.h * w30.s2 + w40.s2) +
                 2.h * (16.h * w00.s3 + 8.h * w10.s3 + 4.h * w20.s3 +
                        2.h * w30.s3 + w40.s3) +
                 (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41)) /
                32400.h;
      out0.s6 = (16.h * (16.h * w00.s0 + 8.h * w10.s0 + 4.h * w20.s0 +
                         2.h * w30.s0 + w40.s0) -
                 8.h * (16.h * w00.s1 + 8.h * w10.s1 + 4.h * w20.s1 +
                        2.h * w30.s1 + w40.s1) +
                 4.h * (16.h * w00.s2 + 8.h * w10.s2 + 4.h * w20.s2 +
                        2.h * w30.s2 + w40.s2) -
                 2.h * (16.h * w00.s3 + 8.h * w10.s3 + 4.h * w20.s3 +
                        2.h * w30.s3 + w40.s3) +
                 (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41)) /
                32400.h;
      out0.s7 = (16.h * w01 + 8.h * w11 + 4.h * w21 + 2.h * w31 + w41) / 180.h;
      break;

    case 6:
      // Row 6
      out0.s0 = (16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 - 2.h * w30.s0 +
                 w40.s0) /
                180.h;
      out0.s1 = -((16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 - 2.h * w30.s0 +
                   w40.s0) +
                  (16.h * w00.s1 - 8.h * w10.s1 + 4.h * w20.s1 - 2.h * w30.s1 +
                   w40.s1) +
                  (16.h * w00.s2 - 8.h * w10.s2 + 4.h * w20.s2 - 2.h * w30.s2 +
                   w40.s2) +
                  (16.h * w00.s3 - 8.h * w10.s3 + 4.h * w20.s3 - 2.h * w30.s3 +
                   w40.s3) +
                  (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41)) /
                810.h;
      out0.s2 = -((16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 - 2.h * w30.s0 +
                   w40.s0) -
                  (16.h * w00.s1 - 8.h * w10.s1 + 4.h * w20.s1 - 2.h * w30.s1 +
                   w40.s1) +
                  (16.h * w00.s2 - 8.h * w10.s2 + 4.h * w20.s2 - 2.h * w30.s2 +
                   w40.s2) -
                  (16.h * w00.s3 - 8.h * w10.s3 + 4.h * w20.s3 - 2.h * w30.s3 +
                   w40.s3) +
                  (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41)) /
                810.h;
      out0.s3 =
          ((16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 - 2.h * w30.s0 +
            w40.s0) +
           2.h * (16.h * w00.s1 - 8.h * w10.s1 + 4.h * w20.s1 - 2.h * w30.s1 +
                  w40.s1) +
           4.h * (16.h * w00.s2 - 8.h * w10.s2 + 4.h * w20.s2 - 2.h * w30.s2 +
                  w40.s2) +
           8.h * (16.h * w00.s3 - 8.h * w10.s3 + 4.h * w20.s3 - 2.h * w30.s3 +
                  w40.s3) +
           16.h * (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41)) /
          16200.h;
      out0.s4 =
          ((16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 - 2.h * w30.s0 +
            w40.s0) -
           2.h * (16.h * w00.s1 - 8.h * w10.s1 + 4.h * w20.s1 - 2.h * w30.s1 +
                  w40.s1) +
           4.h * (16.h * w00.s2 - 8.h * w10.s2 + 4.h * w20.s2 - 2.h * w30.s2 +
                  w40.s2) -
           8.h * (16.h * w00.s3 - 8.h * w10.s3 + 4.h * w20.s3 - 2.h * w30.s3 +
                  w40.s3) +
           16.h * (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41)) /
          16200.h;
      out0.s5 = (16.h * (16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 -
                         2.h * w30.s0 + w40.s0) +
                 8.h * (16.h * w00.s1 - 8.h * w10.s1 + 4.h * w20.s1 -
                        2.h * w30.s1 + w40.s1) +
                 4.h * (16.h * w00.s2 - 8.h * w10.s2 + 4.h * w20.s2 -
                        2.h * w30.s2 + w40.s2) +
                 2.h * (16.h * w00.s3 - 8.h * w10.s3 + 4.h * w20.s3 -
                        2.h * w30.s3 + w40.s3) +
                 (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41)) /
                32400.h;
      out0.s6 = (16.h * (16.h * w00.s0 - 8.h * w10.s0 + 4.h * w20.s0 -
                         2.h * w30.s0 + w40.s0) -
                 8.h * (16.h * w00.s1 - 8.h * w10.s1 + 4.h * w20.s1 -
                        2.h * w30.s1 + w40.s1) +
                 4.h * (16.h * w00.s2 - 8.h * w10.s2 + 4.h * w20.s2 -
                        2.h * w30.s2 + w40.s2) -
                 2.h * (16.h * w00.s3 - 8.h * w10.s3 + 4.h * w20.s3 -
                        2.h * w30.s3 + w40.s3) +
                 (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41)) /
                32400.h;
      out0.s7 = (16.h * w01 - 8.h * w11 + 4.h * w21 - 2.h * w31 + w41) / 180.h;
      break;

    case 7:
      // Row 7
      out0.s0 = w40.s0;
      out0.s1 = -2.h * (w40.s0 + w40.s1 + w40.s2 + w40.s3 + w41) / 9.h;
      out0.s2 = -2.h * (w40.s0 - w40.s1 + w40.s2 - w40.s3 + w41) / 9.h;
      out0.s3 =
          (w40.s0 + 2.h * w40.s1 + 4.h * w40.s2 + 8.h * w40.s3 + 16.h * w41) /
          90.h;
      out0.s4 =
          (w40.s0 - 2.h * w40.s1 + 4.h * w40.s2 - 8.h * w40.s3 + 16.h * w41) /
          90.h;
      out0.s5 =
          (16.h * w40.s0 + 8.h * w40.s1 + 4.h * w40.s2 + 2.h * w40.s3 + w41) /
          180.h;
      out0.s6 =
          (16.h * w40.s0 - 8.h * w40.s1 + 4.h * w40.s2 - 2.h * w40.s3 + w41) /
          180.h;
      out0.s7 = w41;
      break;
    }

    outputs[0] = out0.s0;
    outputs += K * C;
    outputs[0] = out0.s1;
    outputs += K * C;
    outputs[0] = out0.s2;
    outputs += K * C;
    outputs[0] = out0.s3;
    outputs += K * C;
    outputs[0] = out0.s4;
    outputs += K * C;
    outputs[0] = out0.s5;
    outputs += K * C;
    outputs[0] = out0.s6;
    outputs += K * C;
    outputs[0] = out0.s7;
    outputs += K * C;
  }
}

__kernel void conv_wino5_gemm_opt_img(
    __read_only image1d_buffer_t X_A, // X instance of MxK mat
    __read_only image1d_buffer_t X_B, // X instance of KxN mat
    __global half *X_C, int X, int M, int N,
    int K) // X instance of MxN mat
{

  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k, idx;

  // How many output pixels per therad?
  // PT_M x PT_N
  half2 reg_C[PT_M][PT_N_OPT];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N_OPT; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N_OPT;
  int mid = lid / TP_N_OPT;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N

  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {

#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          idx = (toff_m + m + warp_id) * K + (toff_k + k + wid) + (xid * M * K);
          half2 regA =
              SAFE_IMAGE_LOAD_VEC2_HALF(X_A, M, K, (toff_m + m + warp_id),
                                        (toff_k + k + wid), (xid * M * K));

          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              (idx % 2 == 0 ? regA.x : regA.y);
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          idx = (toff_k + k + warp_id) * N + (toff_n + n + wid) + (xid * K * N);
          half2 regB =
              SAFE_IMAGE_LOAD_VEC2_HALF(X_B, K, N, (toff_k + k + warp_id),
                                        (toff_n + n + wid), (xid * K * N));

          (local_B)[((k + warp_id) * TILE_N + n + wid)] =
              (idx % 2 == 0 ? regB.x : regB.y);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
        half2 reg = local_A[(m * TP_M + mid) * TILE_K + k];
#pragma unroll
        for (n = 0; n < PT_N_OPT; n++) {
          idx = k * TILE_N + n * TP_N * 2 + nid * 2;
          half2 regV = vload2(idx >> 1, local_B);
          reg_C[m][n] += reg * regV;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N_OPT; n++) {
      SAFE_VEC2_STORE(C, M, N, (toff_m + m * TP_M + mid),
                      (toff_n + n * TP_N_OPT * 2 + nid * 2), reg_C[m][n]);
    }
  }
}

__kernel void conv_wino5_gemm_opt_img_old(
    __read_only image1d_buffer_t X_A, // X instance of MxK mat
    __read_only image1d_buffer_t X_B, // X instance of KxN mat
    __global half *X_C, int X, int M, int N,
    int K) // X instance of MxN mat
{

  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k;

  // How many output pixels per therad?
  // PT_M x PT_N
  half reg_C[PT_M][PT_N];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N

  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {

#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              SAFE_IMAGE_LOAD_HALF(X_A, M, K, (toff_m + m + warp_id),
                                   (toff_k + k + wid), (xid * M * K));

          // (local_A)[(m + warp_id) * TILE_K + k + wid] =
          // SAFE_IMAGE_LOAD_HALF2(
          //     read_imageh(X_A,(toff_m + m + warp_id) * K + (toff_k + k + wid)
          //     + (xid * M * K)).x, M, K, (toff_m + m + warp_id), (toff_k + k +
          //     wid));

          // (local_A)[(m + warp_id) * TILE_K + k + wid] =
          //   read_imageh(X_A,(toff_m + m + warp_id) * K + (toff_k + k +
          //   wid)).x;
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          (local_B)[((k + warp_id) * TILE_N + n + wid)] =
              SAFE_IMAGE_LOAD_HALF(X_B, K, N, (toff_k + k + warp_id),
                                   (toff_n + n + wid), (xid * K * N));

          // (local_B)[((k + warp_id) * TILE_N + n + wid)] =
          // SAFE_IMAGE_LOAD_HALF2(
          //   read_imageh(X_B,(toff_k + k + warp_id) * N + (toff_n + n + wid) +
          //   (xid * K * N)).x, K, N, (toff_k + k + warp_id), (toff_n + n +
          //   wid));

          // (local_B)[((k + warp_id) * TILE_N + n + wid)] =
          //     read_imageh(X_B, ((toff_k + k + warp_id) * N + (toff_n + n +
          //     wid))).x;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
#pragma unroll
        for (n = 0; n < PT_N; n++) {
          reg_C[m][n] += local_A[(m * TP_M + mid) * TILE_K + k] *
                         local_B[k * TILE_N + n * TP_N + nid];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid),
                 reg_C[m][n]);
    }
  }
}

__kernel void conv_wino5_gemm_opt(__global half *X_A, // X instance of MxK mat
                                  __global half *X_B, // X instance of KxN mat
                                  __global half *X_C, int X, int M, int N,
                                  int K) // X instance of MxN mat
{

  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k;

  // How many output pixels per therad?
  // PT_M x PT_N
  half reg_C[PT_M][PT_N];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *A = X_A + xid * M * K;
  __global half *B = X_B + xid * K * N;
  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {

#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              SAFE_LOAD(A, M, K, (toff_m + m + warp_id), (toff_k + k + wid));
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          (local_B)[((k + warp_id) * TILE_N + n + wid)] =
              SAFE_LOAD(B, K, N, (toff_k + k + warp_id), (toff_n + n + wid));
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
#pragma unroll
        for (n = 0; n < PT_N; n++) {
          reg_C[m][n] += local_A[(m * TP_M + mid) * TILE_K + k] *
                         local_B[k * TILE_N + n * TP_N + nid];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid),
                 reg_C[m][n]);
    }
  }
}

__kernel void conv_wino5_data_untile_no_bias(__global half *inputs,
                                             __global half *outputs, int K,
                                             int H, int W) {

  int TP = (int)ceil((H - 4) / 4.0h);
  int TQ = (int)ceil((W - 4) / 4.0h);

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = bid * lsz + lid;
  int ktptq = gid;
  int k = ktptq / (TP * TQ);
  if (k >= K)
    return;
  int tptq = ktptq - k * (TP * TQ);
  int tp = tptq / (TQ);
  int tq = tptq - tp * (TQ);
  int p = tp * 4, q = tq * 4;

  int P = H - 4;
  int Q = W - 4;

  half m[8][8], TM[4][8];
  half4 M0, M1, M2, M3;

  inputs += (k * TP + tp) * TQ + tq;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      m[i][j] = inputs[0];
      inputs += K * TP * TQ;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    TM[0][i] = 1 * m[0][i] + 1 * m[1][i] + 1 * m[2][i] + 1 * m[3][i] +
               1 * m[4][i] + 8 * m[5][i] + 8 * m[6][i];
    TM[1][i] = 1 * m[1][i] - 1 * m[2][i] + 2 * m[3][i] - 2 * m[4][i] +
               4 * m[5][i] - 4 * m[6][i];
    TM[2][i] = 1 * m[1][i] + 1 * m[2][i] + 4 * m[3][i] + 4 * m[4][i] +
               2 * m[5][i] + 2 * m[6][i];
    TM[3][i] = 1 * m[1][i] - 1 * m[2][i] + 8 * m[3][i] - 8 * m[4][i] +
               1 * m[5][i] - 1 * m[6][i] + 1 * m[7][i];
  }

  M0.s0 = 1 * TM[0][0] + 1 * TM[0][1] + 1 * TM[0][2] + 1 * TM[0][3] +
          1 * TM[0][4] + 8 * TM[0][5] + 8 * TM[0][6];
  M0.s1 = 1 * TM[0][1] - 1 * TM[0][2] + 2 * TM[0][3] - 2 * TM[0][4] +
          4 * TM[0][5] - 4 * TM[0][6];
  M0.s2 = 1 * TM[0][1] + 1 * TM[0][2] + 4 * TM[0][3] + 4 * TM[0][4] +
          2 * TM[0][5] + 2 * TM[0][6];
  M0.s3 = 1 * TM[0][1] - 1 * TM[0][2] + 8 * TM[0][3] - 8 * TM[0][4] +
          1 * TM[0][5] - 1 * TM[0][6] + 1 * TM[0][7];

  M1.s0 = 1 * TM[1][0] + 1 * TM[1][1] + 1 * TM[1][2] + 1 * TM[1][3] +
          1 * TM[1][4] + 8 * TM[1][5] + 8 * TM[1][6];
  M1.s1 = 1 * TM[1][1] - 1 * TM[1][2] + 2 * TM[1][3] - 2 * TM[1][4] +
          4 * TM[1][5] - 4 * TM[1][6];
  M1.s2 = 1 * TM[1][1] + 1 * TM[1][2] + 4 * TM[1][3] + 4 * TM[1][4] +
          2 * TM[1][5] + 2 * TM[1][6];
  M1.s3 = 1 * TM[1][1] - 1 * TM[1][2] + 8 * TM[1][3] - 8 * TM[1][4] +
          1 * TM[1][5] - 1 * TM[1][6] + 1 * TM[1][7];

  M2.s0 = 1 * TM[2][0] + 1 * TM[2][1] + 1 * TM[2][2] + 1 * TM[2][3] +
          1 * TM[2][4] + 8 * TM[2][5] + 8 * TM[2][6];
  M2.s1 = 1 * TM[2][1] - 1 * TM[2][2] + 2 * TM[2][3] - 2 * TM[2][4] +
          4 * TM[2][5] - 4 * TM[2][6];
  M2.s2 = 1 * TM[2][1] + 1 * TM[2][2] + 4 * TM[2][3] + 4 * TM[2][4] +
          2 * TM[2][5] + 2 * TM[2][6];
  M2.s3 = 1 * TM[2][1] - 1 * TM[2][2] + 8 * TM[2][3] - 8 * TM[2][4] +
          1 * TM[2][5] - 1 * TM[2][6] + 1 * TM[2][7];

  M3.s0 = 1 * TM[3][0] + 1 * TM[3][1] + 1 * TM[3][2] + 1 * TM[3][3] +
          1 * TM[3][4] + 8 * TM[3][5] + 8 * TM[3][6];
  M3.s1 = 1 * TM[3][1] - 1 * TM[3][2] + 2 * TM[3][3] - 2 * TM[3][4] +
          4 * TM[3][5] - 4 * TM[3][6];
  M3.s2 = 1 * TM[3][1] + 1 * TM[3][2] + 4 * TM[3][3] + 4 * TM[3][4] +
          2 * TM[3][5] + 2 * TM[3][6];
  M3.s3 = 1 * TM[3][1] - 1 * TM[3][2] + 8 * TM[3][3] - 8 * TM[3][4] +
          1 * TM[3][5] - 1 * TM[3][6] + 1 * TM[3][7];

  outputs += (k * P + p) * Q + q;

  if (p + 3 < P && q + 3 < Q) {
    half *ptr = outputs;
    vstore4(M0, 0, ptr);
    ptr += Q;
    vstore4(M1, 0, ptr);
    ptr += Q;
    vstore4(M2, 0, ptr);
    ptr += Q;
    vstore4(M3, 0, ptr);
  } else {

    if (p + 0 < P && q + 0 < Q)
      outputs[0 * Q + 0] = M0.s0;
    if (p + 0 < P && q + 1 < Q)
      outputs[0 * Q + 1] = M0.s1;
    if (p + 0 < P && q + 2 < Q)
      outputs[0 * Q + 2] = M0.s2;
    if (p + 0 < P && q + 3 < Q)
      outputs[0 * Q + 3] = M0.s3;

    if (p + 1 < P && q + 0 < Q)
      outputs[1 * Q + 0] = M1.s0;
    if (p + 1 < P && q + 1 < Q)
      outputs[1 * Q + 1] = M1.s1;
    if (p + 1 < P && q + 2 < Q)
      outputs[1 * Q + 2] = M1.s2;
    if (p + 1 < P && q + 3 < Q)
      outputs[1 * Q + 3] = M1.s3;

    if (p + 2 < P && q + 0 < Q)
      outputs[2 * Q + 0] = M2.s0;
    if (p + 2 < P && q + 1 < Q)
      outputs[2 * Q + 1] = M2.s1;
    if (p + 2 < P && q + 2 < Q)
      outputs[2 * Q + 2] = M2.s2;
    if (p + 2 < P && q + 3 < Q)
      outputs[2 * Q + 3] = M2.s3;

    if (p + 3 < P && q + 0 < Q)
      outputs[3 * Q + 0] = M3.s0;
    if (p + 3 < P && q + 1 < Q)
      outputs[3 * Q + 1] = M3.s1;
    if (p + 3 < P && q + 2 < Q)
      outputs[3 * Q + 2] = M3.s2;
    if (p + 3 < P && q + 3 < Q)
      outputs[3 * Q + 3] = M3.s3;
  }
}

__kernel void conv_wino5_data_untile(__global half *inputs,
                                     __global half *outputs,
                                     __global half *bias, int K, int H, int W) {

  int TP = (int)ceil((H - 4) / 4.0h);
  int TQ = (int)ceil((W - 4) / 4.0h);

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = bid * lsz + lid;
  int ktptq = gid;
  int k = ktptq / (TP * TQ);
  if (k >= K)
    return;
  int tptq = ktptq - k * (TP * TQ);
  int tp = tptq / (TQ);
  int tq = tptq - tp * (TQ);
  int p = tp * 4, q = tq * 4;

  int P = H - 4;
  int Q = W - 4;

  half m[8][8], TM[4][8];
  half4 M0, M1, M2, M3;

  inputs += (k * TP + tp) * TQ + tq;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      m[i][j] = inputs[0];
      inputs += K * TP * TQ;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    TM[0][i] = 1 * m[0][i] + 1 * m[1][i] + 1 * m[2][i] + 1 * m[3][i] +
               1 * m[4][i] + 8 * m[5][i] + 8 * m[6][i];
    TM[1][i] = 1 * m[1][i] - 1 * m[2][i] + 2 * m[3][i] - 2 * m[4][i] +
               4 * m[5][i] - 4 * m[6][i];
    TM[2][i] = 1 * m[1][i] + 1 * m[2][i] + 4 * m[3][i] + 4 * m[4][i] +
               2 * m[5][i] + 2 * m[6][i];
    TM[3][i] = 1 * m[1][i] - 1 * m[2][i] + 8 * m[3][i] - 8 * m[4][i] +
               1 * m[5][i] - 1 * m[6][i] + 1 * m[7][i];
  }

  M0.s0 = 1 * TM[0][0] + 1 * TM[0][1] + 1 * TM[0][2] + 1 * TM[0][3] +
          1 * TM[0][4] + 8 * TM[0][5] + 8 * TM[0][6];
  M0.s1 = 1 * TM[0][1] - 1 * TM[0][2] + 2 * TM[0][3] - 2 * TM[0][4] +
          4 * TM[0][5] - 4 * TM[0][6];
  M0.s2 = 1 * TM[0][1] + 1 * TM[0][2] + 4 * TM[0][3] + 4 * TM[0][4] +
          2 * TM[0][5] + 2 * TM[0][6];
  M0.s3 = 1 * TM[0][1] - 1 * TM[0][2] + 8 * TM[0][3] - 8 * TM[0][4] +
          1 * TM[0][5] - 1 * TM[0][6] + 1 * TM[0][7];

  M1.s0 = 1 * TM[1][0] + 1 * TM[1][1] + 1 * TM[1][2] + 1 * TM[1][3] +
          1 * TM[1][4] + 8 * TM[1][5] + 8 * TM[1][6];
  M1.s1 = 1 * TM[1][1] - 1 * TM[1][2] + 2 * TM[1][3] - 2 * TM[1][4] +
          4 * TM[1][5] - 4 * TM[1][6];
  M1.s2 = 1 * TM[1][1] + 1 * TM[1][2] + 4 * TM[1][3] + 4 * TM[1][4] +
          2 * TM[1][5] + 2 * TM[1][6];
  M1.s3 = 1 * TM[1][1] - 1 * TM[1][2] + 8 * TM[1][3] - 8 * TM[1][4] +
          1 * TM[1][5] - 1 * TM[1][6] + 1 * TM[1][7];

  M2.s0 = 1 * TM[2][0] + 1 * TM[2][1] + 1 * TM[2][2] + 1 * TM[2][3] +
          1 * TM[2][4] + 8 * TM[2][5] + 8 * TM[2][6];
  M2.s1 = 1 * TM[2][1] - 1 * TM[2][2] + 2 * TM[2][3] - 2 * TM[2][4] +
          4 * TM[2][5] - 4 * TM[2][6];
  M2.s2 = 1 * TM[2][1] + 1 * TM[2][2] + 4 * TM[2][3] + 4 * TM[2][4] +
          2 * TM[2][5] + 2 * TM[2][6];
  M2.s3 = 1 * TM[2][1] - 1 * TM[2][2] + 8 * TM[2][3] - 8 * TM[2][4] +
          1 * TM[2][5] - 1 * TM[2][6] + 1 * TM[2][7];

  M3.s0 = 1 * TM[3][0] + 1 * TM[3][1] + 1 * TM[3][2] + 1 * TM[3][3] +
          1 * TM[3][4] + 8 * TM[3][5] + 8 * TM[3][6];
  M3.s1 = 1 * TM[3][1] - 1 * TM[3][2] + 2 * TM[3][3] - 2 * TM[3][4] +
          4 * TM[3][5] - 4 * TM[3][6];
  M3.s2 = 1 * TM[3][1] + 1 * TM[3][2] + 4 * TM[3][3] + 4 * TM[3][4] +
          2 * TM[3][5] + 2 * TM[3][6];
  M3.s3 = 1 * TM[3][1] - 1 * TM[3][2] + 8 * TM[3][3] - 8 * TM[3][4] +
          1 * TM[3][5] - 1 * TM[3][6] + 1 * TM[3][7];

  outputs += (k * P + p) * Q + q;

  half b = bias[k];
  if (p + 3 < P && q + 3 < Q) {
    half4 B = (half4)(b, b, b, b);

    M0 += B;
    M1 += B;
    M2 += B;
    M3 += B;

    half *ptr = outputs;
    vstore4(M0, 0, ptr);
    ptr += Q;
    vstore4(M1, 0, ptr);
    ptr += Q;
    vstore4(M2, 0, ptr);
    ptr += Q;
    vstore4(M3, 0, ptr);
  } else {

    if (p + 0 < P && q + 0 < Q)
      outputs[0 * Q + 0] = M0.s0;
    if (p + 0 < P && q + 1 < Q)
      outputs[0 * Q + 1] = M0.s1;
    if (p + 0 < P && q + 2 < Q)
      outputs[0 * Q + 2] = M0.s2;
    if (p + 0 < P && q + 3 < Q)
      outputs[0 * Q + 3] = M0.s3;

    if (p + 1 < P && q + 0 < Q)
      outputs[1 * Q + 0] = M1.s0;
    if (p + 1 < P && q + 1 < Q)
      outputs[1 * Q + 1] = M1.s1;
    if (p + 1 < P && q + 2 < Q)
      outputs[1 * Q + 2] = M1.s2;
    if (p + 1 < P && q + 3 < Q)
      outputs[1 * Q + 3] = M1.s3;

    if (p + 2 < P && q + 0 < Q)
      outputs[2 * Q + 0] = M2.s0;
    if (p + 2 < P && q + 1 < Q)
      outputs[2 * Q + 1] = M2.s1;
    if (p + 2 < P && q + 2 < Q)
      outputs[2 * Q + 2] = M2.s2;
    if (p + 2 < P && q + 3 < Q)
      outputs[2 * Q + 3] = M2.s3;

    if (p + 3 < P && q + 0 < Q)
      outputs[3 * Q + 0] = M3.s0;
    if (p + 3 < P && q + 1 < Q)
      outputs[3 * Q + 1] = M3.s1;
    if (p + 3 < P && q + 2 < Q)
      outputs[3 * Q + 2] = M3.s2;
    if (p + 3 < P && q + 3 < Q)
      outputs[3 * Q + 3] = M3.s3;
  }
}

//////////////////////////// vector, register, local memory

__kernel void conv_wino5_data_tile(__global half *inputs,
                                   __global half *outputs, int C, int H,
                                   int W) {

  int PH = H;
  int PW = W;
  int TP = (int)ceil((H - 4) / 4.0h);
  int TQ = (int)ceil((W - 4) / 4.0h);
  int c = get_global_id(2);
  int tp = get_global_id(1);
  int tq = get_global_id(0);
  if (tp >= TP || tq >= TQ)
    return;
  int h = tp * 4;
  int w = tq * 4;
  int lid_tp = get_local_id(1);
  int lid_tq = get_local_id(0);
  int lid_h = lid_tp * 4;
  int lid_w = lid_tq;

  __local half4 v[68][17];

  inputs += (c * PH + h) * PW + w;
  outputs += (c * TP + tp) * TQ + tq; // jump to tile offset to save (output)
  int K = C * TP * TQ;

  // boundary tiles
  if (lid_tp == 15 || lid_tq == 15 || tp == TP - 1 || tq == TQ - 1) {

    // 8x8 load
    half4 v1;
    for (int i = 0; i < 8; i++) {
      v1.x = ((h + i < PH) && (w + 0 < PW)) ? inputs[i * PW + 0] : 0.h;
      v1.y = ((h + i < PH) && (w + 1 < PW)) ? inputs[i * PW + 1] : 0.h;
      v1.z = ((h + i < PH) && (w + 2 < PW)) ? inputs[i * PW + 2] : 0.h;
      v1.w = ((h + i < PH) && (w + 3 < PW)) ? inputs[i * PW + 3] : 0.h;
      v[lid_h + i][lid_w] = v1;
      v1.x = ((h + i < PH) && (w + 4 < PW)) ? inputs[i * PW + 4] : 0.h;
      v1.y = ((h + i < PH) && (w + 5 < PW)) ? inputs[i * PW + 5] : 0.h;
      v1.z = ((h + i < PH) && (w + 6 < PW)) ? inputs[i * PW + 6] : 0.h;
      v1.w = ((h + i < PH) && (w + 7 < PW)) ? inputs[i * PW + 7] : 0.h;
      v[lid_h + i][lid_w + 1] = v1;
    }

  } else {

    // 4x4 load
    for (int i = 0; i < 4; i++) {
      half4 v1 = vload4(0, inputs);
      v[lid_h + i][lid_w] = v1;
      inputs += PW;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  half8 TV, V;

  for (int j = 0; j < 8; j++) {

    switch (j) {
    case 0:
      TV.s0 = 1 * v[lid_h][lid_w].x - 5.25h * v[lid_h + 2][lid_w].x +
              5.25h * v[lid_h + 4][lid_w].x - 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 1 * v[lid_h][lid_w].y - 5.25h * v[lid_h + 2][lid_w].y +
              5.25h * v[lid_h + 4][lid_w].y - 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 1 * v[lid_h][lid_w].z - 5.25h * v[lid_h + 2][lid_w].z +
              5.25h * v[lid_h + 4][lid_w].z - 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 1 * v[lid_h][lid_w].w - 5.25h * v[lid_h + 2][lid_w].w +
              5.25h * v[lid_h + 4][lid_w].w - 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = 1 * v[lid_h][lid_w + 1].x - 5.25h * v[lid_h + 2][lid_w + 1].x +
              5.25h * v[lid_h + 4][lid_w + 1].x - 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = 1 * v[lid_h][lid_w + 1].y - 5.25h * v[lid_h + 2][lid_w + 1].y +
              5.25h * v[lid_h + 4][lid_w + 1].y - 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = 1 * v[lid_h][lid_w + 1].z - 5.25h * v[lid_h + 2][lid_w + 1].z +
              5.25h * v[lid_h + 4][lid_w + 1].z - 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = 1 * v[lid_h][lid_w + 1].w - 5.25h * v[lid_h + 2][lid_w + 1].w +
              5.25h * v[lid_h + 4][lid_w + 1].w - 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 1:
      TV.s0 = 1 * v[lid_h + 1][lid_w].x + 1 * v[lid_h + 2][lid_w].x -
              4.25h * v[lid_h + 3][lid_w].x - 4.25h * v[lid_h + 4][lid_w].x +
              1 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 1 * v[lid_h + 1][lid_w].y + 1 * v[lid_h + 2][lid_w].y -
              4.25h * v[lid_h + 3][lid_w].y - 4.25h * v[lid_h + 4][lid_w].y +
              1 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 1 * v[lid_h + 1][lid_w].z + 1 * v[lid_h + 2][lid_w].z -
              4.25h * v[lid_h + 3][lid_w].z - 4.25h * v[lid_h + 4][lid_w].z +
              1 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 1 * v[lid_h + 1][lid_w].w + 1 * v[lid_h + 2][lid_w].w -
              4.25h * v[lid_h + 3][lid_w].w - 4.25h * v[lid_h + 4][lid_w].w +
              1 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = 1 * v[lid_h + 1][lid_w + 1].x + 1 * v[lid_h + 2][lid_w + 1].x -
              4.25h * v[lid_h + 3][lid_w + 1].x -
              4.25h * v[lid_h + 4][lid_w + 1].x +
              1 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = 1 * v[lid_h + 1][lid_w + 1].y + 1 * v[lid_h + 2][lid_w + 1].y -
              4.25h * v[lid_h + 3][lid_w + 1].y -
              4.25h * v[lid_h + 4][lid_w + 1].y +
              1 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = 1 * v[lid_h + 1][lid_w + 1].z + 1 * v[lid_h + 2][lid_w + 1].z -
              4.25h * v[lid_h + 3][lid_w + 1].z -
              4.25h * v[lid_h + 4][lid_w + 1].z +
              1 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = 1 * v[lid_h + 1][lid_w + 1].w + 1 * v[lid_h + 2][lid_w + 1].w -
              4.25h * v[lid_h + 3][lid_w + 1].w -
              4.25h * v[lid_h + 4][lid_w + 1].w +
              1 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 2:
      TV.s0 = -1 * v[lid_h + 1][lid_w].x + 1 * v[lid_h + 2][lid_w].x +
              4.25h * v[lid_h + 3][lid_w].x - 4.25h * v[lid_h + 4][lid_w].x -
              1 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = -1 * v[lid_h + 1][lid_w].y + 1 * v[lid_h + 2][lid_w].y +
              4.25h * v[lid_h + 3][lid_w].y - 4.25h * v[lid_h + 4][lid_w].y -
              1 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = -1 * v[lid_h + 1][lid_w].z + 1 * v[lid_h + 2][lid_w].z +
              4.25h * v[lid_h + 3][lid_w].z - 4.25h * v[lid_h + 4][lid_w].z -
              1 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = -1 * v[lid_h + 1][lid_w].w + 1 * v[lid_h + 2][lid_w].w +
              4.25h * v[lid_h + 3][lid_w].w - 4.25h * v[lid_h + 4][lid_w].w -
              1 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = -1 * v[lid_h + 1][lid_w + 1].x + 1 * v[lid_h + 2][lid_w + 1].x +
              4.25h * v[lid_h + 3][lid_w + 1].x -
              4.25h * v[lid_h + 4][lid_w + 1].x -
              1 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = -1 * v[lid_h + 1][lid_w + 1].y + 1 * v[lid_h + 2][lid_w + 1].y +
              4.25h * v[lid_h + 3][lid_w + 1].y -
              4.25h * v[lid_h + 4][lid_w + 1].y -
              1 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = -1 * v[lid_h + 1][lid_w + 1].z + 1 * v[lid_h + 2][lid_w + 1].z +
              4.25h * v[lid_h + 3][lid_w + 1].z -
              4.25h * v[lid_h + 4][lid_w + 1].z -
              1 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = -1 * v[lid_h + 1][lid_w + 1].w + 1 * v[lid_h + 2][lid_w + 1].w +
              4.25h * v[lid_h + 3][lid_w + 1].w -
              4.25h * v[lid_h + 4][lid_w + 1].w -
              1 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 3:
      TV.s0 = 0.5h * v[lid_h + 1][lid_w].x + 0.25h * v[lid_h + 2][lid_w].x -
              2.5h * v[lid_h + 3][lid_w].x - 1.25h * v[lid_h + 4][lid_w].x +
              2 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 0.5h * v[lid_h + 1][lid_w].y + 0.25h * v[lid_h + 2][lid_w].y -
              2.5h * v[lid_h + 3][lid_w].y - 1.25h * v[lid_h + 4][lid_w].y +
              2 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 0.5h * v[lid_h + 1][lid_w].z + 0.25h * v[lid_h + 2][lid_w].z -
              2.5h * v[lid_h + 3][lid_w].z - 1.25h * v[lid_h + 4][lid_w].z +
              2 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 0.5h * v[lid_h + 1][lid_w].w + 0.25h * v[lid_h + 2][lid_w].w -
              2.5h * v[lid_h + 3][lid_w].w - 1.25h * v[lid_h + 4][lid_w].w +
              2 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 =
          0.5h * v[lid_h + 1][lid_w + 1].x + 0.25h * v[lid_h + 2][lid_w + 1].x -
          2.5h * v[lid_h + 3][lid_w + 1].x - 1.25h * v[lid_h + 4][lid_w + 1].x +
          2 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 =
          0.5h * v[lid_h + 1][lid_w + 1].y + 0.25h * v[lid_h + 2][lid_w + 1].y -
          2.5h * v[lid_h + 3][lid_w + 1].y - 1.25h * v[lid_h + 4][lid_w + 1].y +
          2 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 =
          0.5h * v[lid_h + 1][lid_w + 1].z + 0.25h * v[lid_h + 2][lid_w + 1].z -
          2.5h * v[lid_h + 3][lid_w + 1].z - 1.25h * v[lid_h + 4][lid_w + 1].z +
          2 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 =
          0.5h * v[lid_h + 1][lid_w + 1].w + 0.25h * v[lid_h + 2][lid_w + 1].w -
          2.5h * v[lid_h + 3][lid_w + 1].w - 1.25h * v[lid_h + 4][lid_w + 1].w +
          2 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 4:
      TV.s0 = -0.5h * v[lid_h + 1][lid_w].x + 0.25h * v[lid_h + 2][lid_w].x +
              2.5h * v[lid_h + 3][lid_w].x - 1.25h * v[lid_h + 4][lid_w].x -
              2 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = -0.5h * v[lid_h + 1][lid_w].y + 0.25h * v[lid_h + 2][lid_w].y +
              2.5h * v[lid_h + 3][lid_w].y - 1.25h * v[lid_h + 4][lid_w].y -
              2 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = -0.5h * v[lid_h + 1][lid_w].z + 0.25h * v[lid_h + 2][lid_w].z +
              2.5h * v[lid_h + 3][lid_w].z - 1.25h * v[lid_h + 4][lid_w].z -
              2 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = -0.5h * v[lid_h + 1][lid_w].w + 0.25h * v[lid_h + 2][lid_w].w +
              2.5h * v[lid_h + 3][lid_w].w - 1.25h * v[lid_h + 4][lid_w].w -
              2 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = -0.5h * v[lid_h + 1][lid_w + 1].x +
              0.25h * v[lid_h + 2][lid_w + 1].x +
              2.5h * v[lid_h + 3][lid_w + 1].x -
              1.25h * v[lid_h + 4][lid_w + 1].x -
              2 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = -0.5h * v[lid_h + 1][lid_w + 1].y +
              0.25h * v[lid_h + 2][lid_w + 1].y +
              2.5h * v[lid_h + 3][lid_w + 1].y -
              1.25h * v[lid_h + 4][lid_w + 1].y -
              2 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = -0.5h * v[lid_h + 1][lid_w + 1].z +
              0.25h * v[lid_h + 2][lid_w + 1].z +
              2.5h * v[lid_h + 3][lid_w + 1].z -
              1.25h * v[lid_h + 4][lid_w + 1].z -
              2 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = -0.5h * v[lid_h + 1][lid_w + 1].w +
              0.25h * v[lid_h + 2][lid_w + 1].w +
              2.5h * v[lid_h + 3][lid_w + 1].w -
              1.25h * v[lid_h + 4][lid_w + 1].w -
              2 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 5:
      TV.s0 = 2 * v[lid_h + 1][lid_w].x + 4 * v[lid_h + 2][lid_w].x -
              2.5h * v[lid_h + 3][lid_w].x - 5.0 * v[lid_h + 4][lid_w].x +
              0.5h * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 2 * v[lid_h + 1][lid_w].y + 4 * v[lid_h + 2][lid_w].y -
              2.5h * v[lid_h + 3][lid_w].y - 5.0 * v[lid_h + 4][lid_w].y +
              0.5h * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 2 * v[lid_h + 1][lid_w].z + 4 * v[lid_h + 2][lid_w].z -
              2.5h * v[lid_h + 3][lid_w].z - 5.0 * v[lid_h + 4][lid_w].z +
              0.5h * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 2 * v[lid_h + 1][lid_w].w + 4 * v[lid_h + 2][lid_w].w -
              2.5h * v[lid_h + 3][lid_w].w - 5.0 * v[lid_h + 4][lid_w].w +
              0.5h * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = 2 * v[lid_h + 1][lid_w + 1].x + 4 * v[lid_h + 2][lid_w + 1].x -
              2.5h * v[lid_h + 3][lid_w + 1].x -
              5.0 * v[lid_h + 4][lid_w + 1].x +
              0.5h * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = 2 * v[lid_h + 1][lid_w + 1].y + 4 * v[lid_h + 2][lid_w + 1].y -
              2.5h * v[lid_h + 3][lid_w + 1].y -
              5.0 * v[lid_h + 4][lid_w + 1].y +
              0.5h * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = 2 * v[lid_h + 1][lid_w + 1].z + 4 * v[lid_h + 2][lid_w + 1].z -
              2.5h * v[lid_h + 3][lid_w + 1].z -
              5.0 * v[lid_h + 4][lid_w + 1].z +
              0.5h * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = 2 * v[lid_h + 1][lid_w + 1].w + 4 * v[lid_h + 2][lid_w + 1].w -
              2.5h * v[lid_h + 3][lid_w + 1].w -
              5.0 * v[lid_h + 4][lid_w + 1].w +
              0.5h * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 6:
      TV.s0 = -2 * v[lid_h + 1][lid_w].x + 4 * v[lid_h + 2][lid_w].x +
              2.5h * v[lid_h + 3][lid_w].x - 5.0 * v[lid_h + 4][lid_w].x -
              0.5h * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = -2 * v[lid_h + 1][lid_w].y + 4 * v[lid_h + 2][lid_w].y +
              2.5h * v[lid_h + 3][lid_w].y - 5.0 * v[lid_h + 4][lid_w].y -
              0.5h * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = -2 * v[lid_h + 1][lid_w].z + 4 * v[lid_h + 2][lid_w].z +
              2.5h * v[lid_h + 3][lid_w].z - 5.0 * v[lid_h + 4][lid_w].z -
              0.5h * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = -2 * v[lid_h + 1][lid_w].w + 4 * v[lid_h + 2][lid_w].w +
              2.5h * v[lid_h + 3][lid_w].w - 5.0 * v[lid_h + 4][lid_w].w -
              0.5h * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = -2 * v[lid_h + 1][lid_w + 1].x + 4 * v[lid_h + 2][lid_w + 1].x +
              2.5h * v[lid_h + 3][lid_w + 1].x -
              5.0 * v[lid_h + 4][lid_w + 1].x -
              0.5h * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = -2 * v[lid_h + 1][lid_w + 1].y + 4 * v[lid_h + 2][lid_w + 1].y +
              2.5h * v[lid_h + 3][lid_w + 1].y -
              5.0 * v[lid_h + 4][lid_w + 1].y -
              0.5h * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = -2 * v[lid_h + 1][lid_w + 1].z + 4 * v[lid_h + 2][lid_w + 1].z +
              2.5h * v[lid_h + 3][lid_w + 1].z -
              5.0 * v[lid_h + 4][lid_w + 1].z -
              0.5h * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = -2 * v[lid_h + 1][lid_w + 1].w + 4 * v[lid_h + 2][lid_w + 1].w +
              2.5h * v[lid_h + 3][lid_w + 1].w -
              5.0 * v[lid_h + 4][lid_w + 1].w -
              0.5h * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 7:
      TV.s0 = -1 * v[lid_h + 1][lid_w].x + 5.25h * v[lid_h + 3][lid_w].x -
              5.25h * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 7][lid_w].x;
      TV.s1 = -1 * v[lid_h + 1][lid_w].y + 5.25h * v[lid_h + 3][lid_w].y -
              5.25h * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 7][lid_w].y;
      TV.s2 = -1 * v[lid_h + 1][lid_w].z + 5.25h * v[lid_h + 3][lid_w].z -
              5.25h * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 7][lid_w].z;
      TV.s3 = -1 * v[lid_h + 1][lid_w].w + 5.25h * v[lid_h + 3][lid_w].w -
              5.25h * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 7][lid_w].w;
      TV.s4 = -1 * v[lid_h + 1][lid_w + 1].x +
              5.25h * v[lid_h + 3][lid_w + 1].x -
              5.25h * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 7][lid_w + 1].x;
      TV.s5 = -1 * v[lid_h + 1][lid_w + 1].y +
              5.25h * v[lid_h + 3][lid_w + 1].y -
              5.25h * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 7][lid_w + 1].y;
      TV.s6 = -1 * v[lid_h + 1][lid_w + 1].z +
              5.25h * v[lid_h + 3][lid_w + 1].z -
              5.25h * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 7][lid_w + 1].z;
      TV.s7 = -1 * v[lid_h + 1][lid_w + 1].w +
              5.25h * v[lid_h + 3][lid_w + 1].w -
              5.25h * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 7][lid_w + 1].w;
      break;
    }

    V.s0 = 1 * TV.s0 - 5.25h * TV.s2 + 5.25h * TV.s4 - 1 * TV.s6;
    V.s1 = 1 * TV.s1 + 1 * TV.s2 - 4.25h * TV.s3 - 4.25h * TV.s4 + 1 * TV.s5 +
           1 * TV.s6;
    V.s2 = -1 * TV.s1 + 1 * TV.s2 + 4.25h * TV.s3 - 4.25h * TV.s4 - 1 * TV.s5 +
           1 * TV.s6;
    V.s3 = 0.5h * TV.s1 + 0.25h * TV.s2 - 2.5h * TV.s3 - 1.25h * TV.s4 +
           2 * TV.s5 + 1 * TV.s6;
    V.s4 = -0.5h * TV.s1 + 0.25h * TV.s2 + 2.5h * TV.s3 - 1.25h * TV.s4 -
           2 * TV.s5 + 1 * TV.s6;
    V.s5 = 2 * TV.s1 + 4 * TV.s2 - 2.5h * TV.s3 - 5.0 * TV.s4 + 0.5h * TV.s5 +
           1 * TV.s6;
    V.s6 = -2 * TV.s1 + 4 * TV.s2 + 2.5h * TV.s3 - 5.0 * TV.s4 - 0.5h * TV.s5 +
           1 * TV.s6;
    V.s7 = -1 * TV.s1 + 5.25h * TV.s3 - 5.25h * TV.s5 + 1 * TV.s7;

    outputs[0] = V.s0;
    outputs += K;
    outputs[0] = V.s1;
    outputs += K;
    outputs[0] = V.s2;
    outputs += K;
    outputs[0] = V.s3;
    outputs += K;
    outputs[0] = V.s4;
    outputs += K;
    outputs[0] = V.s5;
    outputs += K;
    outputs[0] = V.s6;
    outputs += K;
    outputs[0] = V.s7;
    outputs += K;
  }
}

/////////////////////// split conv_wino

__kernel void conv_wino5_data_tile_split(__global half *inputs,
                                         __global half *outputs, int C, int H,
                                         int W, int id) {

  int splitN = 2;
  int PH = H;
  int PW = W;
  int TP = (int)ceil((H - 4) / 4.0h);
  TP /= splitN;
  int TQ = (int)ceil((W - 4) / 4.0h);
  int c = get_global_id(2);
  int tp = get_global_id(1);
  int tq = get_global_id(0);
  if (tp >= TP || tq >= TQ)
    return;
  int h = (tp + TP * id) * 4;
  int w = tq * 4;
  int lid_tp = get_local_id(1);
  int lid_tq = get_local_id(0);
  int lid_h = lid_tp * 4;
  int lid_w = lid_tq;

  inputs += (c * PH + h) * PW + w;
  __local half4 v[68][17];

  outputs += (c * TP + tp) * TQ + tq; // jump to tile offset to save (output)
  int K = C * TP * TQ;

  // boundary tiles
  if (lid_tp == 15 || lid_tq == 15 || tp == TP - 1 || tq == TQ - 1) {

    // 8x8 load
    half4 v1;
    for (int i = 0; i < 8; i++) {
      v1.x = ((h + i < PH) && (w + 0 < PW)) ? inputs[i * PW + 0] : 0.h;
      v1.y = ((h + i < PH) && (w + 1 < PW)) ? inputs[i * PW + 1] : 0.h;
      v1.z = ((h + i < PH) && (w + 2 < PW)) ? inputs[i * PW + 2] : 0.h;
      v1.w = ((h + i < PH) && (w + 3 < PW)) ? inputs[i * PW + 3] : 0.h;
      v[lid_h + i][lid_w] = v1;
      v1.x = ((h + i < PH) && (w + 4 < PW)) ? inputs[i * PW + 4] : 0.h;
      v1.y = ((h + i < PH) && (w + 5 < PW)) ? inputs[i * PW + 5] : 0.h;
      v1.z = ((h + i < PH) && (w + 6 < PW)) ? inputs[i * PW + 6] : 0.h;
      v1.w = ((h + i < PH) && (w + 7 < PW)) ? inputs[i * PW + 7] : 0.h;
      v[lid_h + i][lid_w + 1] = v1;
    }

  } else {

    // 4x4 load
    for (int i = 0; i < 4; i++) {
      half4 v1 = vload4(0, inputs);
      v[lid_h + i][lid_w] = v1;
      inputs += PW;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  half8 TV, V;

  for (int j = 0; j < 8; j++) {

    switch (j) {
    case 0:
      TV.s0 = 1 * v[lid_h][lid_w].x - 5.25h * v[lid_h + 2][lid_w].x +
              5.25h * v[lid_h + 4][lid_w].x - 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 1 * v[lid_h][lid_w].y - 5.25h * v[lid_h + 2][lid_w].y +
              5.25h * v[lid_h + 4][lid_w].y - 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 1 * v[lid_h][lid_w].z - 5.25h * v[lid_h + 2][lid_w].z +
              5.25h * v[lid_h + 4][lid_w].z - 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 1 * v[lid_h][lid_w].w - 5.25h * v[lid_h + 2][lid_w].w +
              5.25h * v[lid_h + 4][lid_w].w - 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = 1 * v[lid_h][lid_w + 1].x - 5.25h * v[lid_h + 2][lid_w + 1].x +
              5.25h * v[lid_h + 4][lid_w + 1].x - 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = 1 * v[lid_h][lid_w + 1].y - 5.25h * v[lid_h + 2][lid_w + 1].y +
              5.25h * v[lid_h + 4][lid_w + 1].y - 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = 1 * v[lid_h][lid_w + 1].z - 5.25h * v[lid_h + 2][lid_w + 1].z +
              5.25h * v[lid_h + 4][lid_w + 1].z - 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = 1 * v[lid_h][lid_w + 1].w - 5.25h * v[lid_h + 2][lid_w + 1].w +
              5.25h * v[lid_h + 4][lid_w + 1].w - 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 1:
      TV.s0 = 1 * v[lid_h + 1][lid_w].x + 1 * v[lid_h + 2][lid_w].x -
              4.25h * v[lid_h + 3][lid_w].x - 4.25h * v[lid_h + 4][lid_w].x +
              1 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 1 * v[lid_h + 1][lid_w].y + 1 * v[lid_h + 2][lid_w].y -
              4.25h * v[lid_h + 3][lid_w].y - 4.25h * v[lid_h + 4][lid_w].y +
              1 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 1 * v[lid_h + 1][lid_w].z + 1 * v[lid_h + 2][lid_w].z -
              4.25h * v[lid_h + 3][lid_w].z - 4.25h * v[lid_h + 4][lid_w].z +
              1 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 1 * v[lid_h + 1][lid_w].w + 1 * v[lid_h + 2][lid_w].w -
              4.25h * v[lid_h + 3][lid_w].w - 4.25h * v[lid_h + 4][lid_w].w +
              1 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = 1 * v[lid_h + 1][lid_w + 1].x + 1 * v[lid_h + 2][lid_w + 1].x -
              4.25h * v[lid_h + 3][lid_w + 1].x -
              4.25h * v[lid_h + 4][lid_w + 1].x +
              1 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = 1 * v[lid_h + 1][lid_w + 1].y + 1 * v[lid_h + 2][lid_w + 1].y -
              4.25h * v[lid_h + 3][lid_w + 1].y -
              4.25h * v[lid_h + 4][lid_w + 1].y +
              1 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = 1 * v[lid_h + 1][lid_w + 1].z + 1 * v[lid_h + 2][lid_w + 1].z -
              4.25h * v[lid_h + 3][lid_w + 1].z -
              4.25h * v[lid_h + 4][lid_w + 1].z +
              1 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = 1 * v[lid_h + 1][lid_w + 1].w + 1 * v[lid_h + 2][lid_w + 1].w -
              4.25h * v[lid_h + 3][lid_w + 1].w -
              4.25h * v[lid_h + 4][lid_w + 1].w +
              1 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 2:
      TV.s0 = -1 * v[lid_h + 1][lid_w].x + 1 * v[lid_h + 2][lid_w].x +
              4.25h * v[lid_h + 3][lid_w].x - 4.25h * v[lid_h + 4][lid_w].x -
              1 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = -1 * v[lid_h + 1][lid_w].y + 1 * v[lid_h + 2][lid_w].y +
              4.25h * v[lid_h + 3][lid_w].y - 4.25h * v[lid_h + 4][lid_w].y -
              1 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = -1 * v[lid_h + 1][lid_w].z + 1 * v[lid_h + 2][lid_w].z +
              4.25h * v[lid_h + 3][lid_w].z - 4.25h * v[lid_h + 4][lid_w].z -
              1 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = -1 * v[lid_h + 1][lid_w].w + 1 * v[lid_h + 2][lid_w].w +
              4.25h * v[lid_h + 3][lid_w].w - 4.25h * v[lid_h + 4][lid_w].w -
              1 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = -1 * v[lid_h + 1][lid_w + 1].x + 1 * v[lid_h + 2][lid_w + 1].x +
              4.25h * v[lid_h + 3][lid_w + 1].x -
              4.25h * v[lid_h + 4][lid_w + 1].x -
              1 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = -1 * v[lid_h + 1][lid_w + 1].y + 1 * v[lid_h + 2][lid_w + 1].y +
              4.25h * v[lid_h + 3][lid_w + 1].y -
              4.25h * v[lid_h + 4][lid_w + 1].y -
              1 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = -1 * v[lid_h + 1][lid_w + 1].z + 1 * v[lid_h + 2][lid_w + 1].z +
              4.25h * v[lid_h + 3][lid_w + 1].z -
              4.25h * v[lid_h + 4][lid_w + 1].z -
              1 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = -1 * v[lid_h + 1][lid_w + 1].w + 1 * v[lid_h + 2][lid_w + 1].w +
              4.25h * v[lid_h + 3][lid_w + 1].w -
              4.25h * v[lid_h + 4][lid_w + 1].w -
              1 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 3:
      TV.s0 = 0.5h * v[lid_h + 1][lid_w].x + 0.25h * v[lid_h + 2][lid_w].x -
              2.5h * v[lid_h + 3][lid_w].x - 1.25h * v[lid_h + 4][lid_w].x +
              2 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 0.5h * v[lid_h + 1][lid_w].y + 0.25h * v[lid_h + 2][lid_w].y -
              2.5h * v[lid_h + 3][lid_w].y - 1.25h * v[lid_h + 4][lid_w].y +
              2 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 0.5h * v[lid_h + 1][lid_w].z + 0.25h * v[lid_h + 2][lid_w].z -
              2.5h * v[lid_h + 3][lid_w].z - 1.25h * v[lid_h + 4][lid_w].z +
              2 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 0.5h * v[lid_h + 1][lid_w].w + 0.25h * v[lid_h + 2][lid_w].w -
              2.5h * v[lid_h + 3][lid_w].w - 1.25h * v[lid_h + 4][lid_w].w +
              2 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 =
          0.5h * v[lid_h + 1][lid_w + 1].x + 0.25h * v[lid_h + 2][lid_w + 1].x -
          2.5h * v[lid_h + 3][lid_w + 1].x - 1.25h * v[lid_h + 4][lid_w + 1].x +
          2 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 =
          0.5h * v[lid_h + 1][lid_w + 1].y + 0.25h * v[lid_h + 2][lid_w + 1].y -
          2.5h * v[lid_h + 3][lid_w + 1].y - 1.25h * v[lid_h + 4][lid_w + 1].y +
          2 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 =
          0.5h * v[lid_h + 1][lid_w + 1].z + 0.25h * v[lid_h + 2][lid_w + 1].z -
          2.5h * v[lid_h + 3][lid_w + 1].z - 1.25h * v[lid_h + 4][lid_w + 1].z +
          2 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 =
          0.5h * v[lid_h + 1][lid_w + 1].w + 0.25h * v[lid_h + 2][lid_w + 1].w -
          2.5h * v[lid_h + 3][lid_w + 1].w - 1.25h * v[lid_h + 4][lid_w + 1].w +
          2 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 4:
      TV.s0 = -0.5h * v[lid_h + 1][lid_w].x + 0.25h * v[lid_h + 2][lid_w].x +
              2.5h * v[lid_h + 3][lid_w].x - 1.25h * v[lid_h + 4][lid_w].x -
              2 * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = -0.5h * v[lid_h + 1][lid_w].y + 0.25h * v[lid_h + 2][lid_w].y +
              2.5h * v[lid_h + 3][lid_w].y - 1.25h * v[lid_h + 4][lid_w].y -
              2 * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = -0.5h * v[lid_h + 1][lid_w].z + 0.25h * v[lid_h + 2][lid_w].z +
              2.5h * v[lid_h + 3][lid_w].z - 1.25h * v[lid_h + 4][lid_w].z -
              2 * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = -0.5h * v[lid_h + 1][lid_w].w + 0.25h * v[lid_h + 2][lid_w].w +
              2.5h * v[lid_h + 3][lid_w].w - 1.25h * v[lid_h + 4][lid_w].w -
              2 * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = -0.5h * v[lid_h + 1][lid_w + 1].x +
              0.25h * v[lid_h + 2][lid_w + 1].x +
              2.5h * v[lid_h + 3][lid_w + 1].x -
              1.25h * v[lid_h + 4][lid_w + 1].x -
              2 * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = -0.5h * v[lid_h + 1][lid_w + 1].y +
              0.25h * v[lid_h + 2][lid_w + 1].y +
              2.5h * v[lid_h + 3][lid_w + 1].y -
              1.25h * v[lid_h + 4][lid_w + 1].y -
              2 * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = -0.5h * v[lid_h + 1][lid_w + 1].z +
              0.25h * v[lid_h + 2][lid_w + 1].z +
              2.5h * v[lid_h + 3][lid_w + 1].z -
              1.25h * v[lid_h + 4][lid_w + 1].z -
              2 * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = -0.5h * v[lid_h + 1][lid_w + 1].w +
              0.25h * v[lid_h + 2][lid_w + 1].w +
              2.5h * v[lid_h + 3][lid_w + 1].w -
              1.25h * v[lid_h + 4][lid_w + 1].w -
              2 * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 5:
      TV.s0 = 2 * v[lid_h + 1][lid_w].x + 4 * v[lid_h + 2][lid_w].x -
              2.5h * v[lid_h + 3][lid_w].x - 5.0 * v[lid_h + 4][lid_w].x +
              0.5h * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = 2 * v[lid_h + 1][lid_w].y + 4 * v[lid_h + 2][lid_w].y -
              2.5h * v[lid_h + 3][lid_w].y - 5.0 * v[lid_h + 4][lid_w].y +
              0.5h * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = 2 * v[lid_h + 1][lid_w].z + 4 * v[lid_h + 2][lid_w].z -
              2.5h * v[lid_h + 3][lid_w].z - 5.0 * v[lid_h + 4][lid_w].z +
              0.5h * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = 2 * v[lid_h + 1][lid_w].w + 4 * v[lid_h + 2][lid_w].w -
              2.5h * v[lid_h + 3][lid_w].w - 5.0 * v[lid_h + 4][lid_w].w +
              0.5h * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = 2 * v[lid_h + 1][lid_w + 1].x + 4 * v[lid_h + 2][lid_w + 1].x -
              2.5h * v[lid_h + 3][lid_w + 1].x -
              5.0 * v[lid_h + 4][lid_w + 1].x +
              0.5h * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = 2 * v[lid_h + 1][lid_w + 1].y + 4 * v[lid_h + 2][lid_w + 1].y -
              2.5h * v[lid_h + 3][lid_w + 1].y -
              5.0 * v[lid_h + 4][lid_w + 1].y +
              0.5h * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = 2 * v[lid_h + 1][lid_w + 1].z + 4 * v[lid_h + 2][lid_w + 1].z -
              2.5h * v[lid_h + 3][lid_w + 1].z -
              5.0 * v[lid_h + 4][lid_w + 1].z +
              0.5h * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = 2 * v[lid_h + 1][lid_w + 1].w + 4 * v[lid_h + 2][lid_w + 1].w -
              2.5h * v[lid_h + 3][lid_w + 1].w -
              5.0 * v[lid_h + 4][lid_w + 1].w +
              0.5h * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 6:
      TV.s0 = -2 * v[lid_h + 1][lid_w].x + 4 * v[lid_h + 2][lid_w].x +
              2.5h * v[lid_h + 3][lid_w].x - 5.0 * v[lid_h + 4][lid_w].x -
              0.5h * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 6][lid_w].x;
      TV.s1 = -2 * v[lid_h + 1][lid_w].y + 4 * v[lid_h + 2][lid_w].y +
              2.5h * v[lid_h + 3][lid_w].y - 5.0 * v[lid_h + 4][lid_w].y -
              0.5h * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 6][lid_w].y;
      TV.s2 = -2 * v[lid_h + 1][lid_w].z + 4 * v[lid_h + 2][lid_w].z +
              2.5h * v[lid_h + 3][lid_w].z - 5.0 * v[lid_h + 4][lid_w].z -
              0.5h * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 6][lid_w].z;
      TV.s3 = -2 * v[lid_h + 1][lid_w].w + 4 * v[lid_h + 2][lid_w].w +
              2.5h * v[lid_h + 3][lid_w].w - 5.0 * v[lid_h + 4][lid_w].w -
              0.5h * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 6][lid_w].w;
      TV.s4 = -2 * v[lid_h + 1][lid_w + 1].x + 4 * v[lid_h + 2][lid_w + 1].x +
              2.5h * v[lid_h + 3][lid_w + 1].x -
              5.0 * v[lid_h + 4][lid_w + 1].x -
              0.5h * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 6][lid_w + 1].x;
      TV.s5 = -2 * v[lid_h + 1][lid_w + 1].y + 4 * v[lid_h + 2][lid_w + 1].y +
              2.5h * v[lid_h + 3][lid_w + 1].y -
              5.0 * v[lid_h + 4][lid_w + 1].y -
              0.5h * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 6][lid_w + 1].y;
      TV.s6 = -2 * v[lid_h + 1][lid_w + 1].z + 4 * v[lid_h + 2][lid_w + 1].z +
              2.5h * v[lid_h + 3][lid_w + 1].z -
              5.0 * v[lid_h + 4][lid_w + 1].z -
              0.5h * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 6][lid_w + 1].z;
      TV.s7 = -2 * v[lid_h + 1][lid_w + 1].w + 4 * v[lid_h + 2][lid_w + 1].w +
              2.5h * v[lid_h + 3][lid_w + 1].w -
              5.0 * v[lid_h + 4][lid_w + 1].w -
              0.5h * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 6][lid_w + 1].w;
      break;

    case 7:
      TV.s0 = -1 * v[lid_h + 1][lid_w].x + 5.25h * v[lid_h + 3][lid_w].x -
              5.25h * v[lid_h + 5][lid_w].x + 1 * v[lid_h + 7][lid_w].x;
      TV.s1 = -1 * v[lid_h + 1][lid_w].y + 5.25h * v[lid_h + 3][lid_w].y -
              5.25h * v[lid_h + 5][lid_w].y + 1 * v[lid_h + 7][lid_w].y;
      TV.s2 = -1 * v[lid_h + 1][lid_w].z + 5.25h * v[lid_h + 3][lid_w].z -
              5.25h * v[lid_h + 5][lid_w].z + 1 * v[lid_h + 7][lid_w].z;
      TV.s3 = -1 * v[lid_h + 1][lid_w].w + 5.25h * v[lid_h + 3][lid_w].w -
              5.25h * v[lid_h + 5][lid_w].w + 1 * v[lid_h + 7][lid_w].w;
      TV.s4 = -1 * v[lid_h + 1][lid_w + 1].x +
              5.25h * v[lid_h + 3][lid_w + 1].x -
              5.25h * v[lid_h + 5][lid_w + 1].x + 1 * v[lid_h + 7][lid_w + 1].x;
      TV.s5 = -1 * v[lid_h + 1][lid_w + 1].y +
              5.25h * v[lid_h + 3][lid_w + 1].y -
              5.25h * v[lid_h + 5][lid_w + 1].y + 1 * v[lid_h + 7][lid_w + 1].y;
      TV.s6 = -1 * v[lid_h + 1][lid_w + 1].z +
              5.25h * v[lid_h + 3][lid_w + 1].z -
              5.25h * v[lid_h + 5][lid_w + 1].z + 1 * v[lid_h + 7][lid_w + 1].z;
      TV.s7 = -1 * v[lid_h + 1][lid_w + 1].w +
              5.25h * v[lid_h + 3][lid_w + 1].w -
              5.25h * v[lid_h + 5][lid_w + 1].w + 1 * v[lid_h + 7][lid_w + 1].w;
      break;
    }

    V.s0 = 1 * TV.s0 - 5.25h * TV.s2 + 5.25h * TV.s4 - 1 * TV.s6;
    V.s1 = 1 * TV.s1 + 1 * TV.s2 - 4.25h * TV.s3 - 4.25h * TV.s4 + 1 * TV.s5 +
           1 * TV.s6;
    V.s2 = -1 * TV.s1 + 1 * TV.s2 + 4.25h * TV.s3 - 4.25h * TV.s4 - 1 * TV.s5 +
           1 * TV.s6;
    V.s3 = 0.5h * TV.s1 + 0.25h * TV.s2 - 2.5h * TV.s3 - 1.25h * TV.s4 +
           2 * TV.s5 + 1 * TV.s6;
    V.s4 = -0.5h * TV.s1 + 0.25h * TV.s2 + 2.5h * TV.s3 - 1.25h * TV.s4 -
           2 * TV.s5 + 1 * TV.s6;
    V.s5 = 2 * TV.s1 + 4 * TV.s2 - 2.5h * TV.s3 - 5.0 * TV.s4 + 0.5h * TV.s5 +
           1 * TV.s6;
    V.s6 = -2 * TV.s1 + 4 * TV.s2 + 2.5h * TV.s3 - 5.0 * TV.s4 - 0.5h * TV.s5 +
           1 * TV.s6;
    V.s7 = -1 * TV.s1 + 5.25h * TV.s3 - 5.25h * TV.s5 + 1 * TV.s7;

    outputs[0] = V.s0;
    outputs += K;
    outputs[0] = V.s1;
    outputs += K;
    outputs[0] = V.s2;
    outputs += K;
    outputs[0] = V.s3;
    outputs += K;
    outputs[0] = V.s4;
    outputs += K;
    outputs[0] = V.s5;
    outputs += K;
    outputs[0] = V.s6;
    outputs += K;
    outputs[0] = V.s7;
    outputs += K;
  }
}

__kernel void conv_wino5_gemm_opt_split_img(
    __read_only image1d_buffer_t X_A, // X instance of MxK mat
    __read_only image1d_buffer_t X_B, // X instance of KxN mat
    __global half *X_C, int X, int M, int N,
    int K) // X instance of MxN mat
{

  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k, idx;

  // How many output pixels per therad?
  // PT_M x PT_N
  half2 reg_C[PT_M][PT_N_OPT];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N_OPT; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N_OPT;
  int mid = lid / TP_N_OPT;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {

#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          idx = (toff_m + m + warp_id) * K + (toff_k + k + wid) + (xid * M * K);

          half2 regA =
              SAFE_IMAGE_LOAD_VEC2_HALF(X_A, M, K, (toff_m + m + warp_id),
                                        (toff_k + k + wid), (xid * M * K));

          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              (idx % 2 == 0 ? regA.x : regA.y);
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {

          idx = (toff_k + k + warp_id) * N + (toff_n + n + wid) + (xid * K * N);

          half2 regB =
              SAFE_IMAGE_LOAD_VEC2_HALF(X_B, K, N, (toff_k + k + warp_id),
                                        (toff_n + n + wid), (xid * K * N));

          (local_B)[((k + warp_id) * TILE_N + n + wid)] =
              (idx % 2 == 0 ? regB.x : regB.y);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
        half2 reg = local_A[(m * TP_M + mid) * TILE_K + k];
#pragma unroll
        for (n = 0; n < PT_N_OPT; n++) {
          idx = k * TILE_N + n * TP_N * 2 + nid * 2;
          half2 regV = vload2(idx >> 1, local_B);
          reg_C[m][n] += reg * regV;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N_OPT; n++) {
      SAFE_VEC2_STORE(C, M, N, (toff_m + m * TP_M + mid),
                      (toff_n + n * TP_N_OPT * 2 + nid * 2), reg_C[m][n]);
    }
  }
}

__kernel void conv_wino5_gemm_opt_split_img_old(
    __read_only image1d_buffer_t X_A, // X instance of MxK mat
    __read_only image1d_buffer_t X_B, // X instance of KxN mat
    __global half *X_C, int X, int M, int N,
    int K) // X instance of MxN mat
{

  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k;

  // How many output pixels per therad?
  // PT_M x PT_N
  half reg_C[PT_M][PT_N];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {

#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              SAFE_IMAGE_LOAD_HALF(X_A, M, K, (toff_m + m + warp_id),
                                   (toff_k + k + wid), (xid * M * K));

          // (local_A)[(m + warp_id) * TILE_K + k + wid] =
          // SAFE_IMAGE_LOAD_HALF2(
          //     read_imageh(X_A,(toff_m + m + warp_id) * K + (toff_k + k + wid)
          //     + (xid * M * K)).x, M, K, (toff_m + m + warp_id), (toff_k + k +
          //     wid));

          // (local_A)[(m + warp_id) * TILE_K + k + wid] =
          //   read_imageh(X_A,(toff_m + m + warp_id) * K + (toff_k + k +
          //   wid)).x;
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          (local_B)[((k + warp_id) * TILE_N + n + wid)] =
              SAFE_IMAGE_LOAD_HALF(X_B, K, N, (toff_k + k + warp_id),
                                   (toff_n + n + wid), (xid * K * N));

          // (local_B)[((k + warp_id) * TILE_N + n + wid)] =
          // SAFE_IMAGE_LOAD_HALF2(
          //   read_imageh(X_B,(toff_k + k + warp_id) * N + (toff_n + n + wid) +
          //   (xid * K * N)).x, K, N, (toff_k + k + warp_id), (toff_n + n +
          //   wid));

          // (local_B)[((k + warp_id) * TILE_N + n + wid)] =
          //     read_imageh(X_B, ((toff_k + k + warp_id) * N + (toff_n + n +
          //     wid))).x;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
#pragma unroll
        for (n = 0; n < PT_N; n++) {
          reg_C[m][n] += local_A[(m * TP_M + mid) * TILE_K + k] *
                         local_B[k * TILE_N + n * TP_N + nid];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid),
                 reg_C[m][n]);
    }
  }
}

__kernel void
conv_wino5_gemm_opt_split(__global half *X_A, // X instance of MxK mat
                          __global half *X_B, // X instance of KxN mat
                          __global half *X_C, int X, int M, int N,
                          int K) // X instance of MxN mat
{

  // For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (get_group_id(0));
  int lid = (get_local_id(0)); // 0 <= lid < BSZ

  int warp_id = lid / WARP_SIZE;
  int wid = lid % WARP_SIZE;

  int m, n, k;

  // How many output pixels per therad?
  // PT_M x PT_N
  half reg_C[PT_M][PT_N];
#pragma unroll
  for (int i = 0; i < PT_M; i++)
    for (int j = 0; j < PT_N; j++)
      reg_C[i][j] = 0;
  // per block?
  // TILE_M x TILE_N
  // How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;

  // How many blocks per instance?
  // Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;

  if (xid > X)
    return;

  // TILE_M x TILE_K x TILE_N
  __local half local_A[TILE_M * TILE_K];
  __local half local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  // N / TILE_N
  __global half *A = X_A + xid * M * K;
  __global half *B = X_B + xid * K * N;
  __global half *C = X_C + xid * M * N;

  for (toff_k = 0; toff_k < K; toff_k += TILE_K) {
    {

#pragma unroll
      for (m = 0; m < TILE_M; m += WARP_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += WARP_SIZE) {
          (local_A)[(m + warp_id) * TILE_K + k + wid] =
              SAFE_LOAD(A, M, K, (toff_m + m + warp_id), (toff_k + k + wid));
        }
      }

#pragma unroll
      for (k = 0; k < TILE_K; k += WARP_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += WARP_SIZE) {
          (local_B)[((k + warp_id) * TILE_N + n + wid)] =
              SAFE_LOAD(B, K, N, (toff_k + k + warp_id), (toff_n + n + wid));
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Block multiplication
#pragma unroll
    for (k = 0; k < TILE_K; k++) {
#pragma unroll
      for (m = 0; m < PT_M; m++) {
#pragma unroll
        for (n = 0; n < PT_N; n++) {
          reg_C[m][n] += local_A[(m * TP_M + mid) * TILE_K + k] *
                         local_B[k * TILE_N + n * TP_N + nid];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (m = 0; m < PT_M; m++) {
#pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid),
                 reg_C[m][n]);
    }
  }
}

__kernel void conv_wino5_data_untile_no_bias_split(__global half *inputs,
                                                   __global half *outputs,
                                                   int K, int H, int W,
                                                   int id) {

  int TP = (int)ceil((H - 4) / 4.0h) / 2.h;
  int TQ = (int)ceil((W - 4) / 4.0h);

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = bid * lsz + lid;
  int ktptq = gid;
  int k = ktptq / (TP * TQ);
  if (k >= K)
    return;
  int tptq = ktptq - k * (TP * TQ);
  int tp = tptq / (TQ);
  int tq = tptq - tp * (TQ);

  int p = (tp + TP * id) * 4;
  int q = tq * 4;

  int P = H - 4;
  int Q = W - 4;

  half m[8][8], TM[4][8];
  half4 M0, M1, M2, M3;

  inputs += (k * TP + tp) * TQ + tq;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      m[i][j] = inputs[0];
      inputs += K * TP * TQ;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    TM[0][i] = 1 * m[0][i] + 1 * m[1][i] + 1 * m[2][i] + 1 * m[3][i] +
               1 * m[4][i] + 8 * m[5][i] + 8 * m[6][i];
    TM[1][i] = 1 * m[1][i] - 1 * m[2][i] + 2 * m[3][i] - 2 * m[4][i] +
               4 * m[5][i] - 4 * m[6][i];
    TM[2][i] = 1 * m[1][i] + 1 * m[2][i] + 4 * m[3][i] + 4 * m[4][i] +
               2 * m[5][i] + 2 * m[6][i];
    TM[3][i] = 1 * m[1][i] - 1 * m[2][i] + 8 * m[3][i] - 8 * m[4][i] +
               1 * m[5][i] - 1 * m[6][i] + 1 * m[7][i];
  }

  M0.s0 = 1 * TM[0][0] + 1 * TM[0][1] + 1 * TM[0][2] + 1 * TM[0][3] +
          1 * TM[0][4] + 8 * TM[0][5] + 8 * TM[0][6];
  M0.s1 = 1 * TM[0][1] - 1 * TM[0][2] + 2 * TM[0][3] - 2 * TM[0][4] +
          4 * TM[0][5] - 4 * TM[0][6];
  M0.s2 = 1 * TM[0][1] + 1 * TM[0][2] + 4 * TM[0][3] + 4 * TM[0][4] +
          2 * TM[0][5] + 2 * TM[0][6];
  M0.s3 = 1 * TM[0][1] - 1 * TM[0][2] + 8 * TM[0][3] - 8 * TM[0][4] +
          1 * TM[0][5] - 1 * TM[0][6] + 1 * TM[0][7];

  M1.s0 = 1 * TM[1][0] + 1 * TM[1][1] + 1 * TM[1][2] + 1 * TM[1][3] +
          1 * TM[1][4] + 8 * TM[1][5] + 8 * TM[1][6];
  M1.s1 = 1 * TM[1][1] - 1 * TM[1][2] + 2 * TM[1][3] - 2 * TM[1][4] +
          4 * TM[1][5] - 4 * TM[1][6];
  M1.s2 = 1 * TM[1][1] + 1 * TM[1][2] + 4 * TM[1][3] + 4 * TM[1][4] +
          2 * TM[1][5] + 2 * TM[1][6];
  M1.s3 = 1 * TM[1][1] - 1 * TM[1][2] + 8 * TM[1][3] - 8 * TM[1][4] +
          1 * TM[1][5] - 1 * TM[1][6] + 1 * TM[1][7];

  M2.s0 = 1 * TM[2][0] + 1 * TM[2][1] + 1 * TM[2][2] + 1 * TM[2][3] +
          1 * TM[2][4] + 8 * TM[2][5] + 8 * TM[2][6];
  M2.s1 = 1 * TM[2][1] - 1 * TM[2][2] + 2 * TM[2][3] - 2 * TM[2][4] +
          4 * TM[2][5] - 4 * TM[2][6];
  M2.s2 = 1 * TM[2][1] + 1 * TM[2][2] + 4 * TM[2][3] + 4 * TM[2][4] +
          2 * TM[2][5] + 2 * TM[2][6];
  M2.s3 = 1 * TM[2][1] - 1 * TM[2][2] + 8 * TM[2][3] - 8 * TM[2][4] +
          1 * TM[2][5] - 1 * TM[2][6] + 1 * TM[2][7];

  M3.s0 = 1 * TM[3][0] + 1 * TM[3][1] + 1 * TM[3][2] + 1 * TM[3][3] +
          1 * TM[3][4] + 8 * TM[3][5] + 8 * TM[3][6];
  M3.s1 = 1 * TM[3][1] - 1 * TM[3][2] + 2 * TM[3][3] - 2 * TM[3][4] +
          4 * TM[3][5] - 4 * TM[3][6];
  M3.s2 = 1 * TM[3][1] + 1 * TM[3][2] + 4 * TM[3][3] + 4 * TM[3][4] +
          2 * TM[3][5] + 2 * TM[3][6];
  M3.s3 = 1 * TM[3][1] - 1 * TM[3][2] + 8 * TM[3][3] - 8 * TM[3][4] +
          1 * TM[3][5] - 1 * TM[3][6] + 1 * TM[3][7];

  outputs += (k * P + p) * Q + q;

  if (p + 3 < P && q + 3 < Q) {
    half *ptr = outputs;
    vstore4(M0, 0, ptr);
    ptr += Q;
    vstore4(M1, 0, ptr);
    ptr += Q;
    vstore4(M2, 0, ptr);
    ptr += Q;
    vstore4(M3, 0, ptr);
  } else {

    if (p + 0 < P && q + 0 < Q)
      outputs[0 * Q + 0] = M0.s0;
    if (p + 0 < P && q + 1 < Q)
      outputs[0 * Q + 1] = M0.s1;
    if (p + 0 < P && q + 2 < Q)
      outputs[0 * Q + 2] = M0.s2;
    if (p + 0 < P && q + 3 < Q)
      outputs[0 * Q + 3] = M0.s3;

    if (p + 1 < P && q + 0 < Q)
      outputs[1 * Q + 0] = M1.s0;
    if (p + 1 < P && q + 1 < Q)
      outputs[1 * Q + 1] = M1.s1;
    if (p + 1 < P && q + 2 < Q)
      outputs[1 * Q + 2] = M1.s2;
    if (p + 1 < P && q + 3 < Q)
      outputs[1 * Q + 3] = M1.s3;

    if (p + 2 < P && q + 0 < Q)
      outputs[2 * Q + 0] = M2.s0;
    if (p + 2 < P && q + 1 < Q)
      outputs[2 * Q + 1] = M2.s1;
    if (p + 2 < P && q + 2 < Q)
      outputs[2 * Q + 2] = M2.s2;
    if (p + 2 < P && q + 3 < Q)
      outputs[2 * Q + 3] = M2.s3;

    if (p + 3 < P && q + 0 < Q)
      outputs[3 * Q + 0] = M3.s0;
    if (p + 3 < P && q + 1 < Q)
      outputs[3 * Q + 1] = M3.s1;
    if (p + 3 < P && q + 2 < Q)
      outputs[3 * Q + 2] = M3.s2;
    if (p + 3 < P && q + 3 < Q)
      outputs[3 * Q + 3] = M3.s3;
  }
}

__kernel void conv_wino5_data_untile_split(__global half *inputs,
                                           __global half *outputs,
                                           __global half *bias, int K, int H,
                                           int W, int id) {

  int TP = (int)ceil((H - 4) / 4.0h) / 2;
  int TQ = (int)ceil((W - 4) / 4.0h);

  int lid = get_local_id(0);
  int lsz = get_local_size(0);
  int bid = get_group_id(0);
  int gid = bid * lsz + lid;
  int ktptq = gid;
  int k = ktptq / (TP * TQ);
  if (k >= K)
    return;
  int tptq = ktptq - k * (TP * TQ);
  int tp = tptq / (TQ);
  int tq = tptq - tp * (TQ);

  int p = (tp + TP * id) * 4;
  int q = tq * 4;

  int P = H - 4;
  int Q = W - 4;

  half m[8][8], TM[4][8];
  half4 M0, M1, M2, M3;

  inputs += (k * TP + tp) * TQ + tq;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      m[i][j] = inputs[0];
      inputs += K * TP * TQ;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    TM[0][i] = 1 * m[0][i] + 1 * m[1][i] + 1 * m[2][i] + 1 * m[3][i] +
               1 * m[4][i] + 8 * m[5][i] + 8 * m[6][i];
    TM[1][i] = 1 * m[1][i] - 1 * m[2][i] + 2 * m[3][i] - 2 * m[4][i] +
               4 * m[5][i] - 4 * m[6][i];
    TM[2][i] = 1 * m[1][i] + 1 * m[2][i] + 4 * m[3][i] + 4 * m[4][i] +
               2 * m[5][i] + 2 * m[6][i];
    TM[3][i] = 1 * m[1][i] - 1 * m[2][i] + 8 * m[3][i] - 8 * m[4][i] +
               1 * m[5][i] - 1 * m[6][i] + 1 * m[7][i];
  }

  M0.s0 = 1 * TM[0][0] + 1 * TM[0][1] + 1 * TM[0][2] + 1 * TM[0][3] +
          1 * TM[0][4] + 8 * TM[0][5] + 8 * TM[0][6];
  M0.s1 = 1 * TM[0][1] - 1 * TM[0][2] + 2 * TM[0][3] - 2 * TM[0][4] +
          4 * TM[0][5] - 4 * TM[0][6];
  M0.s2 = 1 * TM[0][1] + 1 * TM[0][2] + 4 * TM[0][3] + 4 * TM[0][4] +
          2 * TM[0][5] + 2 * TM[0][6];
  M0.s3 = 1 * TM[0][1] - 1 * TM[0][2] + 8 * TM[0][3] - 8 * TM[0][4] +
          1 * TM[0][5] - 1 * TM[0][6] + 1 * TM[0][7];

  M1.s0 = 1 * TM[1][0] + 1 * TM[1][1] + 1 * TM[1][2] + 1 * TM[1][3] +
          1 * TM[1][4] + 8 * TM[1][5] + 8 * TM[1][6];
  M1.s1 = 1 * TM[1][1] - 1 * TM[1][2] + 2 * TM[1][3] - 2 * TM[1][4] +
          4 * TM[1][5] - 4 * TM[1][6];
  M1.s2 = 1 * TM[1][1] + 1 * TM[1][2] + 4 * TM[1][3] + 4 * TM[1][4] +
          2 * TM[1][5] + 2 * TM[1][6];
  M1.s3 = 1 * TM[1][1] - 1 * TM[1][2] + 8 * TM[1][3] - 8 * TM[1][4] +
          1 * TM[1][5] - 1 * TM[1][6] + 1 * TM[1][7];

  M2.s0 = 1 * TM[2][0] + 1 * TM[2][1] + 1 * TM[2][2] + 1 * TM[2][3] +
          1 * TM[2][4] + 8 * TM[2][5] + 8 * TM[2][6];
  M2.s1 = 1 * TM[2][1] - 1 * TM[2][2] + 2 * TM[2][3] - 2 * TM[2][4] +
          4 * TM[2][5] - 4 * TM[2][6];
  M2.s2 = 1 * TM[2][1] + 1 * TM[2][2] + 4 * TM[2][3] + 4 * TM[2][4] +
          2 * TM[2][5] + 2 * TM[2][6];
  M2.s3 = 1 * TM[2][1] - 1 * TM[2][2] + 8 * TM[2][3] - 8 * TM[2][4] +
          1 * TM[2][5] - 1 * TM[2][6] + 1 * TM[2][7];

  M3.s0 = 1 * TM[3][0] + 1 * TM[3][1] + 1 * TM[3][2] + 1 * TM[3][3] +
          1 * TM[3][4] + 8 * TM[3][5] + 8 * TM[3][6];
  M3.s1 = 1 * TM[3][1] - 1 * TM[3][2] + 2 * TM[3][3] - 2 * TM[3][4] +
          4 * TM[3][5] - 4 * TM[3][6];
  M3.s2 = 1 * TM[3][1] + 1 * TM[3][2] + 4 * TM[3][3] + 4 * TM[3][4] +
          2 * TM[3][5] + 2 * TM[3][6];
  M3.s3 = 1 * TM[3][1] - 1 * TM[3][2] + 8 * TM[3][3] - 8 * TM[3][4] +
          1 * TM[3][5] - 1 * TM[3][6] + 1 * TM[3][7];

  outputs += (k * P + p) * Q + q;

  half b = bias[k];
  if (p + 3 < P && q + 3 < Q) {
    half4 B = (half4)(b, b, b, b);

    M0 += B;
    M1 += B;
    M2 += B;
    M3 += B;

    half *ptr = outputs;
    vstore4(M0, 0, ptr);
    ptr += Q;
    vstore4(M1, 0, ptr);
    ptr += Q;
    vstore4(M2, 0, ptr);
    ptr += Q;
    vstore4(M3, 0, ptr);
  } else {

    if (p + 0 < P && q + 0 < Q)
      outputs[0 * Q + 0] = M0.s0;
    if (p + 0 < P && q + 1 < Q)
      outputs[0 * Q + 1] = M0.s1;
    if (p + 0 < P && q + 2 < Q)
      outputs[0 * Q + 2] = M0.s2;
    if (p + 0 < P && q + 3 < Q)
      outputs[0 * Q + 3] = M0.s3;

    if (p + 1 < P && q + 0 < Q)
      outputs[1 * Q + 0] = M1.s0;
    if (p + 1 < P && q + 1 < Q)
      outputs[1 * Q + 1] = M1.s1;
    if (p + 1 < P && q + 2 < Q)
      outputs[1 * Q + 2] = M1.s2;
    if (p + 1 < P && q + 3 < Q)
      outputs[1 * Q + 3] = M1.s3;

    if (p + 2 < P && q + 0 < Q)
      outputs[2 * Q + 0] = M2.s0;
    if (p + 2 < P && q + 1 < Q)
      outputs[2 * Q + 1] = M2.s1;
    if (p + 2 < P && q + 2 < Q)
      outputs[2 * Q + 2] = M2.s2;
    if (p + 2 < P && q + 3 < Q)
      outputs[2 * Q + 3] = M2.s3;

    if (p + 3 < P && q + 0 < Q)
      outputs[3 * Q + 0] = M3.s0;
    if (p + 3 < P && q + 1 < Q)
      outputs[3 * Q + 1] = M3.s1;
    if (p + 3 < P && q + 2 < Q)
      outputs[3 * Q + 2] = M3.s2;
    if (p + 3 < P && q + 3 < Q)
      outputs[3 * Q + 3] = M3.s3;
  }
}
