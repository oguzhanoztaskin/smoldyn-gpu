/*
 * data_providers.cu
 *
 *  Created on: Jun 3, 2010
 *      Author: denis
 */

#include "color_map.h"

#include <cuda_gl_interop.h>

namespace	dsmc
{

void	gpu_data_provider_t::init(uint w, uint h)
{
	colorBuffer = CreateCUDABufferObject(w*h*4*sizeof(char), true);
}

void	gpu_data_provider_t::lock_data(uint width, uint height)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, colorBuffer);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE, 0);
}

void	gpu_data_provider_t::unlock_data()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void	gpu_data_provider_t::deinit()
{
	cudaGLUnregisterBufferObject(colorBuffer);
	glDeleteBuffers(1, &colorBuffer);
}

}
