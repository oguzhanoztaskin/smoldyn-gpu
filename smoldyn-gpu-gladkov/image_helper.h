/*
 * image_helper.h
 *
 *  Created on: Jun 4, 2010
 *      Author: denis
 */

#ifndef IMAGE_HELPER_H_
#define IMAGE_HELPER_H_

#include "bitmap.h"
#include "config.h"

void save_png(const char* filename, unsigned char* data, uint w, uint h,
              uint bpp);
bool load_png(const char* filename, bitmap_t& bmp);

void draw_line(unsigned char* data, uint width, uint x0, uint y0, uint x1,
               uint y1, unsigned char color);

#endif /* IMAGE_HELPER_H_ */
