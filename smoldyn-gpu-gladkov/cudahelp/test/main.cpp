#include <iostream>

#include "multi_gpu.h"
#include "streams_gpu.h"
#include "test_rngs.h"

int main(int argc, char** argv) {
  try {
    //	return	do_multi_gpu_test(argc, argv);
    //	return	do_gpu_streams_test(argc, argv);
    test_fast_rngs_gpu();
    test_rngs_gpu();
  } catch (std::exception& e) {
    std::cerr << "Got an exception: " << e.what() << "\n";
  }
}
