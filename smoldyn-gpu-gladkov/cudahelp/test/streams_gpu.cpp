#include "streams_gpu.h"

#include "../benchmark.h"

#include "../devmanager.h"

#include <iostream>

#include "kernels.h"

#include "utils.h"

#include <string.h>

using namespace	std;

using namespace cudahelp;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


bool	verify_data(int*	in, int*	out, int size)
{
	for(int i = 0; i < size; i++)
		if(out[i]/in[i] != 2)
			return false;

	return true;
}

void	init_data(int*	&in, int*	&out, int size)
{
	HANDLE_ERROR(cudaHostAlloc(&in, size*sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc(&out, size*sizeof(int), cudaHostAllocDefault));

	generate_random(in, size);
}

void	free_data(int*	in, int*	out)
{
	cudaFreeHost(in);
	cudaFreeHost(out);
}

void	test_streams(int*	in, int* out, unsigned int size, unsigned int bChunkSize)
{
	DeviceManager::DevicePtr dev = DeviceManager::Get().GetMaxGFpsDevice();
	dev->SetCurrent();

	if(!dev->SupportOverlaps())
		throw	std::runtime_error("Device doesn't support ovelapping");

	const	unsigned int	chunkSize = bChunkSize / 4;
	const	unsigned int	chunksCount = size / chunkSize;


	std::cout<<"Data size: "<<size/1024/1024*4<<" Mb Chunk size: "<<bChunkSize/1024/1024<<" Mb "<<" Elements count "<<chunkSize<<" Chunk count: "<<chunksCount<<"\n";

	int*	in0, *out0;
	int*	in1, *out1;

	cudaStream_t	stream0, stream1;

	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	cudaMalloc(&in0, bChunkSize);
	cudaMalloc(&out0, bChunkSize);
	cudaMalloc(&in1, bChunkSize);
	cudaMalloc(&out1, bChunkSize);

	int*	pIn = in;
	int*	pOut = out;

	for(int i = 0; i < chunksCount/2; i++)
	{
		cudaMemcpyAsync(in0, pIn, bChunkSize, cudaMemcpyHostToDevice, stream0);
		pIn += chunkSize;

		cudaMemcpyAsync(in1, pIn, bChunkSize, cudaMemcpyHostToDevice, stream1);
		pIn += chunkSize;

		MultiplyGPUStreams(in0, out0, chunkSize, stream0);
		MultiplyGPUStreams(in1, out1, chunkSize, stream1);

		cudaMemcpyAsync(pOut, out0, bChunkSize, cudaMemcpyDeviceToHost, stream0);
		pOut += chunkSize;

		cudaMemcpyAsync(pOut, out1, bChunkSize, cudaMemcpyDeviceToHost, stream1);
		pOut += chunkSize;
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaThreadSynchronize();

	cudaFree(in0);
	cudaFree(out0);
	cudaFree(in1);
	cudaFree(out1);

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
}

void	test_no_streams(int*	in, int* out, unsigned int size)
{
	DeviceManager::DevicePtr dev = DeviceManager::Get().GetMaxGFpsDevice();
	dev->SetCurrent();

	int*	dev_in, *dev_out;

	int		bSize = size*sizeof(int);

	cudaMalloc(&dev_in, bSize);
	cudaMalloc(&dev_out, bSize);

	cudaMemcpy(dev_in, in, bSize, cudaMemcpyHostToDevice);

	MultiplyGPU(dev_in, dev_out, size);

	cudaMemcpy(out, dev_out, bSize, cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);		
}

int		do_gpu_streams_test(int	argc, char**	argv)
{
	bool	useStreams = false;

	unsigned int	size = 0x10000000>>3;

	unsigned int	bChunkSize = 1024*1024;

	for(int i = 1; i < argc; i++)
	{
		if(!strcmp("-streams", argv[i]))
			useStreams = true;

		if(!strcmp("-count", argv[i]))
			size = atoi(argv[i+1]);

		if(!strcmp("-chunkSize", argv[i]))
			bChunkSize = atoi(argv[i+1]);
	}

	cout<<"Use streams: "<<bool2str(useStreams)<<" Array size: "<<size<<"\n";

	CpuClockBenchmark	bench;

	int*	in_data, *out_data;

	init_data(in_data, out_data, size);

	try
	{
		bench.Start("StreamsTest");
		if(useStreams)
			test_streams(in_data, out_data, size, bChunkSize);
		else
			test_no_streams(in_data, out_data, size);
		bench.Stop("StreamsTest");

		std::cout<<"Verifying data...\n";

		if(verify_data(in_data, out_data, size))
			cout<<"TEST PASSED\n";
		else
			cout<<"TEST FAILED\n";

	}
	catch(std::exception& e)
	{
		std::cerr<<"Exception: "<<e.what()<<"\n";
	}

	free_data(in_data, out_data);

	bench.Print(cout);

	return 0;
}
