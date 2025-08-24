/*
 * bound_cond.cuh
 *
 *  Created on: Nov 3, 2010
 *      Author: denis
 */

#ifndef BOUND_COND_CUH_
#define BOUND_COND_CUH_

#include "molecules_type.h"

using namespace smolgpu;

struct ReflectiveBC {
  ReflectiveBC(float mn, float mx) : val_min(mn), val_max(mx) {}

  __device__ bool process(float& old, bool& exists) {
    if (old >= val_max) {
      old = 2 * val_max - old;
      return true;
    } else if (old <= val_min) {
      old = 2 * val_min - old;
      return true;
    }

    return false;
  }

  float val_max;
  float val_min;
};

struct PeriodicBC {
  PeriodicBC(float mn, float mx) : val_min(mn), val_max(mx) {}

  __device__ bool process(float& v, bool& exists) {
    if (v >= val_max) {
      v = val_min + 0.01f;
      return true;
    }

    if (v <= val_min) {
      v = val_max - 0.01f;
      return true;
    }

    return false;
  }

  float val_max;
  float val_min;
};

struct TransparentBC {
  __device__ bool process(float& old, bool& exists) { return false; }
};

struct AbsorbBC {
  AbsorbBC(float mn, float mx) : val_min(mn), val_max(mx) {}

  __device__ bool process(float& old, bool& exists) {
    if (old >= val_max) {
      exists = false;
      return true;
    } else if (old <= val_min) {
      exists = false;
      return true;
    }

    return false;
  }

  float val_max;
  float val_min;
};

// a,r,p,t
// r,p
const char* bc_names[] = {"rrr", "rrp", "rpr", "rpp", "prr",
                          "prp", "ppr", "ppp", "aaa", "000"};

template <class CondX, class CondY, class CondZ>
__global__ void ProcessBoundaryConditionsDev(smolgpu::molecule_t* mol,
                                             int count, CondX condx,
                                             CondY condy, CondZ condz) {
  uint idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= count) return;

  smolgpu::molecule_t m = mol[idx];

  bool exist = true;

  bool wx = condx.process(m.pos.x, exist);
  bool wy = condy.process(m.pos.y, exist);
  bool wz = condz.process(m.pos.z, exist);

  if (!exist) m.type = -1;

  if (wx || wy || wz) mol[idx] = m;
}

void ProcessRRR(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, ReflectiveBC(bmin.x, bmax.x), ReflectiveBC(bmin.y, bmax.y),
      ReflectiveBC(bmin.z, bmax.z));
}

void ProcessAAA(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, AbsorbBC(bmin.x, bmax.x), AbsorbBC(bmin.y, bmax.y),
      AbsorbBC(bmin.z, bmax.z));
}

void ProcessRRP(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, ReflectiveBC(bmin.x, bmax.x), ReflectiveBC(bmin.y, bmax.y),
      PeriodicBC(bmin.z, bmax.z));
}

void ProcessRPR(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, ReflectiveBC(bmin.x, bmax.x), PeriodicBC(bmin.y, bmax.y),
      ReflectiveBC(bmin.z, bmax.z));
}

void ProcessRPP(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, ReflectiveBC(bmin.x, bmax.x), PeriodicBC(bmin.y, bmax.y),
      PeriodicBC(bmin.z, bmax.z));
}

void ProcessPRR(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, PeriodicBC(bmin.x, bmax.x), ReflectiveBC(bmin.y, bmax.y),
      ReflectiveBC(bmin.z, bmax.z));
}

void ProcessPRP(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, PeriodicBC(bmin.x, bmax.x), ReflectiveBC(bmin.y, bmax.y),
      PeriodicBC(bmin.z, bmax.z));
}

void ProcessPPR(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, PeriodicBC(bmin.x, bmax.x), PeriodicBC(bmin.y, bmax.y),
      ReflectiveBC(bmin.z, bmax.z));
}

void ProcessPPP(molecule_t* mols, int size, float3 bmin, float3 bmax,
                int blocks, int threads) {
  ProcessBoundaryConditionsDev<<<blocks, threads>>>(
      mols, size, PeriodicBC(bmin.x, bmax.x), PeriodicBC(bmin.y, bmax.y),
      PeriodicBC(bmin.z, bmax.z));
}

typedef void (*bc_func)(molecule_t*, int, float3, float3, int, int);

bc_func bc_functions[] = {ProcessRRR, ProcessRRP, ProcessRPR,
                          ProcessRPP, ProcessPRR, ProcessPRP,
                          ProcessPPR, ProcessPPP, ProcessAAA};

__host__ void ProcessBoundaryConditionsLookup(molecule_t* mols, int size,
                                              float3 mn, float3 mx,
                                              char* fname) {
  int numThreads = 128;
  int numBlocks = GetNumberOfBlocks(numThreads, size);

  int i = 0;

  while (strncmp(bc_names[i], "000", 3)) {
    if (strncmp(bc_names[i], fname, 3) == 0) {
      bc_functions[i](mols, size, mn, mx, numBlocks, numThreads);
      break;
    }
    ++i;
  }
}

#endif /* BOUND_COND_CUH_ */
