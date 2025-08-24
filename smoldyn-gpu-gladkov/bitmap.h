/*
 * bitmap.h
 *
 *  Created on: Nov 9, 2010
 *      Author: denis
 */

#ifndef BITMAP_H_
#define BITMAP_H_

struct	bitmap_t
{
	int		width;
	int		height;
	//bytes per pixel
	int		bpp;
	unsigned char*	pixels;

	bitmap_t(): width(0), height(0), bpp(0), pixels(0)
	{
	}

	~bitmap_t()
	{
		if(pixels)
			delete [] pixels;
	}

	int	size() {return height*width*bpp;}

	void	resize(int w, int h,  int b)
	{
		width = w;
		height = h;
		bpp = b;

		pixels = new unsigned char[size()];
	}
};

#endif /* BITMAP_H_ */
