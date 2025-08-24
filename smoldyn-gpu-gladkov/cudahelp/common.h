#ifndef	UTILS_H
#define	UTILS_H

namespace	cudahelp
{
	int	GetNumberOfBlocks(int	numThreads, int	numSamples);
	bool	CheckCUDAError(const char *msg);
	void	CheckCUDAErrorAndThrow(const char *msg);
}

#endif
