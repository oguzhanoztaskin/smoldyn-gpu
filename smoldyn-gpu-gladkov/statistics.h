/*
 * statistics.h
 *
 *  Created on: Jun 2, 2010
 *      Author: denis
 */

#ifndef STATISTICS_H_
#define STATISTICS_H_

#include "config.h"

namespace	dsmc
{
	struct	statistics_t
	{
		float	numInSample;
		float3	xyzSum;
		float	velMag;
		float2	padd;
	};
}

#endif /* STATISTICS_H_ */
