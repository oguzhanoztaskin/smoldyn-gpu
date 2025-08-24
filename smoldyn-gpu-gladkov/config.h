/*
 *  config.h
 *  
 *
 *  Created by Denis Gladkov on 2/9/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef	CONFIG_H
#define CONFIG_H
/*
#include <vector_types.h>
#include <vector_functions.h>
*/
typedef	unsigned int	uint;
typedef unsigned char	byte;

//#define USE_TEX
#define GPU_TEST
#define USE_GRAPHICS

inline	uint	GetNumberOfBlocks(uint threads, uint numParticles)
{
	uint	numBlocks = numParticles/threads;

	if(numParticles%threads)
		numBlocks++;

	return numBlocks;
}


#ifdef __CDT_PARSER__
#define __host__
#define __constant__
#define __global__
#define __device__
#define __shared__
#define __align__

#endif

#endif
