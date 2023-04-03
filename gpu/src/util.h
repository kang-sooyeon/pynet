
#ifndef UTIL_H_
#define UTIL_H_

#include <CL/cl.h>
#include <memory.h>
#include <pngLoader.h>

#include <iostream>
#include <vector>

#include "half.hpp"

using namespace std;
using half_float::half;
using namespace half_float::literal;


#define DATA_TYPE half

#define CheckCl(exp)                                                   \
  do {                                                                 \
    cl_int status = (exp);                                             \
    if (status != CL_SUCCESS) {                                        \
      fprintf(stderr, "[%s] Error on line %d: (code=%d) \n", __FILE__, \
              __LINE__, static_cast<int>(status));                     \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

inline void CheckBuildProgram(cl_device_id device, cl_program program,
                              cl_int err) {
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t logSize;
    char *log;
    CheckCl(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                  NULL, &logSize));
    log = (char *)malloc(logSize + 1);
    CheckCl(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  logSize, log, NULL));
    log[logSize] = '\0';
    cout << "Compiler error: " << log << endl;
    free(log);
    exit(0);
  } else if (err != 0) {
    fprintf(stderr, "[%s] Build Warning on line %d :  \n", __FILE__, __LINE__);
  }
}

class Data {
 public:
  int c;
  int k;
  int h;
  int w;
  int r;
  int cnt;

  Data(int _c, int _k, int _h, int _w, int _r, int _cnt) {
    c = _c;
    k = _k;
    h = _h;
    w = _w;
    r = _r;
    cnt = _cnt;
  }
};

void Print(char *layerName, gpu::Memory *m);
void PrintWithSave(char *layerName, gpu::Memory *m);

void Add(gpu::Memory *input, gpu::Memory *output);

void ConvertFromFp16(gpu::Memory *input, gpu::Memory *output, int C, int H,
                     int W);
void ConvertToFp16(gpu::Memory *input, gpu::Memory *output, int C, int H,
                   int W);

static double get_time();
void timer_start(int i);
double timer_stop(int i);

void SavePng(void *array, size_t size, int idx, int H, int W);

float *load(string filepath, size_t &num_elements_);
bool compare(gpu::Memory *input, gpu::Memory *target);
void checkError(string path, gpu::Memory *m);

void write(string filepath, gpu::Memory *m);
void psnr(string path1, string path2);
#endif
