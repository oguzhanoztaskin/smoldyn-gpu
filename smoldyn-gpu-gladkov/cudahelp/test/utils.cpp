#include "utils.h"

#include <algorithm>

void generate_random(int* data, int size) {
  std::generate(data, data + size, int_rand);
}