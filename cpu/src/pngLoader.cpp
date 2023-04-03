

#include "pngLoader.h"

void abort_(const char *s, ...) {
  va_list args;
  va_start(args, s);
  vfprintf(stderr, s, args);
  fprintf(stderr, "\n");
  va_end(args);
  abort();
}

PngLoader::PngLoader() {}

void PngLoader::ReadFile(char *fileName) {
  char header[8];  // 8 is the maximum size that can be checked

  /* open file and test for it being a png */
  FILE *fp = fopen(fileName, "rb");
  if (!fp)
    abort_("[read_png_file] File %s could not be opened for reading", fileName);
  fread(header, 1, 8, fp);
  if (png_sig_cmp((png_const_bytep)header, 0, 8))
    abort_("[read_png_file] File %s is not recognized as a PNG file", fileName);

  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr) abort_("[read_png_file] png_create_read_struct failed");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) abort_("[read_png_file] png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[read_png_file] Error during init_io");

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  /// 칼라타입에 따라 channel 세팅
  // channel =
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  if (setjmp(png_jmpbuf(png_ptr)))
    abort_("[read_png_file] Error during read_image");

  row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++)
    row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));

  png_read_image(png_ptr, row_pointers);

  fclose(fp);
}

void PngLoader::PrintFile(void) {
  for (int y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for (int x = 0; x < width; x++) {
      // png_byte *ptr = &(row[x * 4]);
      /*
      printf("Pixel at position [ %d - %d ] has RGB values: %d - %d - %d\n",
                      x, y, ptr[0], ptr[1], ptr[2]);
      */

      png_byte *ptr = &(row[x * 2]);
      short val = 0;
      val = val | (ptr[0] << 8);
      val = val | ptr[1];

      printf(
          "Pixel at position [ %d - %d ] has 16 bit gray scale values: "
          "%d(%x)\n",
          x, y, val, val);
    }
  }
}

/*
float *PngLoader::Ptr(void)
{
        //float *data = new float[width * height];
        float *data = new float[width * height];
        int idx = 0;
        for (int y = 0; y < height; y++)
        {
                png_byte *row = row_pointers[y];
                for (int x = 0; x < width; x++)
                {
                        png_byte *ptr = &(row[x * 2]);
                        short val = 0;
                        val = val | (ptr[0] << 8);
                        val = val | ptr[1];
                        data[idx++] = (float)val;
                }
        }
        return data;
}
*/

size_t *PngLoader::Ptr(void) {
  size_t *data = (size_t *)malloc(width * height * sizeof(size_t));
  int idx = 0;
  for (int y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for (int x = 0; x < width; x++) {
      png_byte *ptr = &(row[x * 2]);
      short val = 0;
      val = val | (ptr[0] << 8);
      val = val | ptr[1];
      data[idx++] = (size_t)val;
    }
  }
  return data;
}

int PngLoader::Size() { return width * height * sizeof(size_t); }

int PngLoader::Width() { return width; }

int PngLoader::Height() { return height; }

PngLoader::~PngLoader() {
  // free(data); ??
}

/*
int main(int argc, char **argv)
{
        if (argc != 2)
                abort_("Usage: program_name <file_in> <file_out>");

        PngLoader pngLoader;
        pngLoader.ReadFile(argv[1]);
        pngLoader.PrintFile();

        return 0;
}
*/
