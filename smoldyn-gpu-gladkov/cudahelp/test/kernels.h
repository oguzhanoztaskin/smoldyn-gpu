#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
void MultiplyInplaceGPU(int* data, int size);
void MultiplyGPU(int* in_data, int* out_data, int size);
void MultiplyGPUStreams(int* in_data, int* out_data, int size, cudaStream_t s);
void TestMersenneTwister(int* data, int count, int rngs);
void TestMersenneTwisterClass(int* data, int count, int rngs);
void TestFastRngs(int* data, int count, int rngs);
void TestFastRngsClass(int* data, int count, int rngs);
}

#endif