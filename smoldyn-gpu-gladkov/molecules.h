/*
 * particles.h
 *
 *  Created on: Jun 1, 2010
 *      Author: denis
 */

#ifndef MOLECULES_H_
#define MOLECULES_H_

#include <algorithm>

#include "config.h"
#include "smol_kernels.cuh"

namespace smolgpu {

struct base_mols_t {
  uint maxMolecules;
  uint aliveMolecules;
  uint* isAlive;

  base_mols_t() : maxMolecules(0), aliveMolecules(0), isAlive(0) {}
  ~base_mols_t();

  void init(uint maxSamp);
};

struct molecule_state_t : public base_mols_t {
  float3* colors;
  molecule_t* molecules;

  molecule_state_t() : molecules(0), colors(0) {}
  ~molecule_state_t();

  float3* lock_colors() { return colors; }
  void unlock_colors() {}

  molecule_t* lock_molecules() { return molecules; }
  void unlock_molecules() {}

  void init(uint samples);

  void render();
};

struct molecule_state_vbo_t : public base_mols_t {
  uint moleculesVBO;
  uint colorsVBO;

  molecule_t* prev_mols;

  molecule_state_vbo_t() : moleculesVBO(0), colorsVBO(0), prev_mols(0) {}
  ~molecule_state_vbo_t();

  float3* lock_colors();
  void unlock_colors();

  molecule_t* lock_molecules();
  void unlock_molecules();

  void copy_prev_mols(molecule_t* curr = 0);

  void init(uint samples);

  void render();
};

}  // namespace smolgpu

#endif /* PARTICLES_H_ */
