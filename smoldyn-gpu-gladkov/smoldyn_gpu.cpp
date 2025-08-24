/*
 * smoldyn_gpu.cpp
 *
 *  Created on: Oct 26, 2010
 *      Author: denis
 */

#include "smolparser.h"

using namespace smolgpu;

int main11(int argc, char **argv) {
  smolparams_t params;

  SmoldynParser parser;
  parser(argv[1], params);

  return 0;
}
