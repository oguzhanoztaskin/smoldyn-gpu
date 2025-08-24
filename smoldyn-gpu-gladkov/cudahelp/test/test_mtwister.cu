#include "../random/mtwister.cuh"

#include "kernels.h"

#include "../common.h"

__global__	void	TestMersenneTwisterGPU(int*	data, int count)
{
	const	int		tid = blockDim.x*blockIdx.x + threadIdx.x;
	const	int	numThreads = blockDim.x*gridDim.x;

	cudahelp::rand::mt_state_t	state = cudahelp::rand::GetMTState(tid);
	cudahelp::rand::mt_struct_stripped_t	config = cudahelp::rand::GetMTConfig(tid);

	for(int idx = tid; idx < count; idx += numThreads)
		data[idx] = cudahelp::rand::MTwisterRndInt(state, config);

	cudahelp::rand::SaveMTState(tid, state);
}       

__global__	void	TestMersenneTwisterClassGPU(int*	data, int count)
{
	int		idx = blockDim.x*blockIdx.x + threadIdx.x;
	const	int	numThreads = blockDim.x*gridDim.x;

	cudahelp::rand::MTGenerator	gen(idx);

	for(; idx < count; idx += numThreads)
		data[idx] = gen.GetInt();
}

void	TestMersenneTwister(int*	data, int count, int rngs)
{
	int	numThreads = 256;
	int	numBlocks = cudahelp::GetNumberOfBlocks(numThreads, rngs);

	TestMersenneTwisterGPU<<<numBlocks, numThreads>>>(data, count);

	cudaThreadSynchronize();

	cudahelp::CheckCUDAError("TestMersenneTwisterGPU");

}

void	TestMersenneTwisterClass(int*	data, int count, int rngs)
{
	int	numThreads = 256;
	int	numBlocks = cudahelp::GetNumberOfBlocks(numThreads, rngs);

	TestMersenneTwisterClassGPU<<<numBlocks, numThreads>>>(data, count);

	cudaThreadSynchronize();

	cudahelp::CheckCUDAError("TestMersenneTwisterClassGPU");
}
