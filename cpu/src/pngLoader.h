#ifndef PNGLOADER_H_
#define PNGLOADER_H_

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>

#define PNG_DEBUG 3
#include <png.h>
#include <zlib.h>

class PngLoader {
 private:
  int width, height, channel;
  png_byte color_type;
  png_byte bit_depth;

  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep *row_pointers;

 public:
  PngLoader();
  ~PngLoader();
  void ReadFile(char *fileName);
  void PrintFile(void);
  // float* Ptr();
  size_t *Ptr();
  int Width();
  int Height();
  int Size();
};

#endif
