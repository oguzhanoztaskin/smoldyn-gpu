/*
 * image_helper.cpp
 *
 *  Created on: Jun 4, 2010
 *      Author: denis
 */

#include "image_helper.h"

#include <math.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "config.h"

void save_png(const char* filename, unsigned char* data, uint w, uint h,
              uint bpp) {
#ifndef _MSC_VER
  /* create file */
  FILE* fp = fopen(filename, "wb");

  if (!fp)
    throw std::runtime_error("DSMCSolver::SaveColorMap. Can't open file");

  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    throw std::runtime_error(
        "DSMCSolver::SaveColorMap. png_create_write_struct failed");

  png_infop info_ptr = png_create_info_struct(png_ptr);

  if (!info_ptr)
    throw std::runtime_error(
        "DSMCSolver::SaveColorMap. png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    throw std::runtime_error("DSMCSolver::SaveColorMap. Error during init_io");

  png_init_io(png_ptr, fp);

  if (setjmp(png_jmpbuf(png_ptr)))
    throw std::runtime_error(
        "DSMCSolver::SaveColorMap. Error during writing header");

  int color_type;

  switch (bpp) {
    case 1:
      color_type = PNG_COLOR_TYPE_GRAY;
      break;
    case 4:
      color_type = PNG_COLOR_TYPE_RGB_ALPHA;
      break;
    default:
      break;
  }

  png_set_IHDR(png_ptr, info_ptr, w, h, 8, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  /* write bytes */
  if (setjmp(png_jmpbuf(png_ptr)))
    throw std::runtime_error(
        "DSMCSolver::SaveColorMap. Error during writing bytes");

  png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * h);

  for (uint i = 0; i < h; i++)
    row_pointers[h - i - 1] = (png_bytep)&data[i * w * bpp];

  png_write_image(png_ptr, row_pointers);

  /* end write */
  if (setjmp(png_jmpbuf(png_ptr)))
    throw std::runtime_error(
        "DSMCSolver::SaveColorMap. Error during end of write");

  png_write_end(png_ptr, NULL);

  free(row_pointers);

  fclose(fp);
#endif
}

#define PNGSIGSIZE 8

bool load_png(const char* filename, bitmap_t& bmp) {
  FILE* fp = fopen(filename, "rb");

  if (!fp) {
    std::cerr << "Error: can't open file " << filename << "\n";
    exit(EXIT_FAILURE);
  }

  /* Optional: Make sure it's a PNG. */
  char header[8];
  fread(header, 1, 8, fp);
  if (png_sig_cmp((png_byte*)header, 0, 8) != 0) {
    return false;
  }

  /* Those NULLs are: error_ptr, error_fn and warn_fn. */
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    return false;
  }

  /* libpng's exception-style error handling using setjmp/longjmp.
     Return here in the event of an error.
     No special headers needed. */
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    if (fp) {
      fclose(fp);
    }
    return false;
  }

  png_init_io(png_ptr, fp);

  /* Skip this if you didn't check the header earlier. */
  png_set_sig_bytes(png_ptr, 8);

  /* No, this really doesn't return anything. */
  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  fclose(fp);
  fp = NULL;

  /* Getting basic image info. */
  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);
  int bpp = png_get_channels(png_ptr, info_ptr);

  bmp.resize(width, height, bpp);

  /* Get IMAGE DATA. Comes in the form of an array of pointers to pixel rows. */
  png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

  unsigned char* bp = bmp.pixels;

  for (uint i = 0; i < height; i++) {
    memcpy(bp, row_pointers[i],
           width * bpp);  // row_pointers[i] = (png_bytep)&data[i*w*bpp];
    bp += width * bpp;
  }

  //	png_destroy_read_struct(&png_ptr, &info_ptr,(png_infopp)0);

  delete[] row_pointers;

  return true;
}

inline void putPixel(unsigned char* data, uint width, uint x, uint y,
                     char col) {
  data[y * width + x] = col;
}

void draw_line(unsigned char* data, uint width, uint x0, uint y0, uint x1,
               uint y1, unsigned char color) {
  int dy = y1 - y0;
  int dx = x1 - x0;
  float t = (float)0.5;  // offset for rounding

  putPixel(data, width, x0, y0, color);

  if (abs(dx) > abs(dy)) {            // slope < 1
    float m = (float)dy / (float)dx;  // compute slope
    t += y0;
    dx = (dx < 0) ? -1 : 1;
    m *= dx;
    while (x0 != x1) {
      x0 += dx;  // step to next x value
      t += m;    // add slope to y value

      putPixel(data, width, x0, (uint)t, color);
    }
  } else {                            // slope >= 1
    float m = (float)dx / (float)dy;  // compute slope
    t += x0;
    dy = (dy < 0) ? -1 : 1;
    m *= dy;

    while (y0 != y1) {
      y0 += dy;  // step to next y value
      t += m;    // add slope to x value

      putPixel(data, width, (uint)t, y0, color);
    }
  }
}
