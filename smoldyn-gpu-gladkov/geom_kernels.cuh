/*
 * geom_kernels.cuh
 *
 *  Created on: Feb 21, 2011
 *      Author: denis
 */

#ifndef GEOM_KERNELS_CUH_
#define GEOM_KERNELS_CUH_

#include "model.h"
#include "molecules_type.h"

#ifdef __gnu_linux__
#include <boost/shared_ptr.hpp>
#else
#include <memory>
#endif

namespace smolgpu {
struct collision_grid_t {
  float3 mmin;
  float3 mmax;
  float3 cellSize;
  uint3 dim;
};

struct geom_desc_t {
  uint3* devIndices;
  uint2* devIndicesStartEnd;
  float3* devVertices;
  float3* devVerticesWorld;
  float3* devFaceNormals;
  uint* devHashes;
  int vertSize;
  int facesSize;
};

struct geom_node_t {
  collision_grid_t grid;

  float4 matrix[4];

  geom_desc_t geom;

  bool moved;

  int type;
  float prob;

  void transform(Frame3D& f);

  geom_node_t() : moved(false), type(0), prob(0) {
    geom.devIndices = 0;
    geom.devIndicesStartEnd = 0;
    geom.devVertices = 0;
    geom.devFaceNormals = 0;
    geom.devHashes = 0;
    geom.devVerticesWorld = 0;
  }

  ~geom_node_t();
};

#ifdef __gnu_linux__  // cuda in linux doesn't work normally with std=c++0x
typedef boost::shared_ptr<geom_node_t> geom_node_ptr;
#else
typedef std::shared_ptr<geom_node_t> geom_node_ptr;
#endif

namespace kernels {
__host__ void CreateGeometryNodeDev(geom_node_t& node, Model& model);
__host__ void ProcessCollisions(geom_node_t& node, molecule_t* curr,
                                molecule_t* prev, int size);
__host__ void TransformGeometry(geom_node_t& node);
}  // namespace kernels
}  // namespace smolgpu

#endif /* GEOM_KERNELS_CUH_ */
