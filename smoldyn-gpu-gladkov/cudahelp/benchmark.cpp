/*
 * benchmarker.cpp
 *
 *  Created on: Aug 15, 2010
 *      Author: denis
 */

#include "benchmark.h"

#include <time.h>

namespace	cudahelp
{

//Use only for GPU clocking!
CudaTimerEvents::CudaTimerEvents()
{
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
}

CudaTimerEvents::~CudaTimerEvents()
{
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
}

void	CudaTimerEvents::start()
{
	cudaEventRecord( startEvent, 0 );
}

float	CudaTimerEvents::stop()
{
	float	gpuTime = 0;
	cudaEventRecord( stopEvent, 0 );
	cudaEventSynchronize( stopEvent );
	cudaEventElapsedTime( &gpuTime, startEvent, stopEvent );

	return gpuTime;
}

ClockTimer::ClockTimer()
{
}

ClockTimer::~ClockTimer()
{
}

void	ClockTimer::start()
{
	time = clock();
}

float	ClockTimer::stop()
{
	return ((clock() - time)/(float)CLOCKS_PER_SEC)*1000.0f;
}
}