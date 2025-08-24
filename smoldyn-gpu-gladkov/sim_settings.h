#ifndef SIM_SETTINGS_H
#define SIM_SETTINGS_H

#include <string>

#include "aabb.h"
#include "config.h"
#include "gas_props.h"

struct settings_t {
  bool benchmark;
  bool cpu;
  bool geometry;
  bool mt;

  bool shaders;

  bool csv;

  uint numParticles;
  uint partPerCell;

  float maxRunTime;

  float dt;

  dsmc::gas_props_t gas;

  std::string gas_name;
  std::string modelName;
  std::string statName;
  std::string concName;
  std::string velsName;

  float density;
  float fnum;
  float temp;

  float viscValue;

  vec3_t streamVelocity;

  char periodicCondition;

  bbox_t boundaries;
  vec3_t cell_size;
  vec3i_t grid_dim;

  uint getCellsCount() const { return grid_dim.x * grid_dim.y * grid_dim.z; }

  void readFromFile(const char* filename);
  void readFromCmdLine(int argc, char** argv);
  void print();

  settings_t();
};

#endif
