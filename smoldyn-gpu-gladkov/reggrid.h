/*
 * reggrid.h
 *
 *  Created on: Mar 16, 2010
 *      Author: denis
 */

#ifndef REGGRID_H_
#define REGGRID_H_

#include <vector>

#include "aabb.h"
#include "frame3d.h"
#include "vector.h"

struct reg_grid_t {
  bbox_t gridSize;
  vec3_t cellSize;
  vec3i_t dim;

  typedef std::vector<uint> idx_vector_t;
  typedef std::vector<vec3_t> normals_vector_t;
  typedef std::vector<normals_vector_t> normals_t;
  typedef std::pair<idx_vector_t, bool> node_t;
  typedef std::vector<node_t> nodes_t;

  nodes_t nodes;
  normals_t normals;

  void reset(const bbox_t& size, const vec3i_t& dm);
  AABB_t getAABB(const vec3i_t& idx);
  uint getCellIdx(const vec3i_t& idx);
  uint cellCount() { return dim.x * dim.y * dim.z; }

  bbox_t getSizeTransformed(Frame3D& f) {
    bbox_t ns;

    matrix4x4 m = f.GetMatrix();

    ns.min = m * gridSize.min;
    ns.max = m * gridSize.max;

    return ns;
  }

  vec3_t getCellSizeTransformed(Frame3D& f) {
    vec3_t ns;

    vec3_t s = f.GetScale();

    ns.x = s.x * cellSize.x;
    ns.y = s.y * cellSize.y;
    ns.z = s.z * cellSize.z;

    return ns;
  }

  void transform(Frame3D& f) {
    matrix4x4 m = f.GetMatrix();

    gridSize.min = m * gridSize.min;
    gridSize.max = m * gridSize.max;

    vec3_t gridWidth = gridSize.max - gridSize.min;

    cellSize.x = gridWidth.x / dim.x;
    cellSize.y = gridWidth.y / dim.y;
    cellSize.z = gridWidth.z / dim.z;
  }

  bool needClosure(uint x, uint y);

  void encloseGrid();

  node_t& operator[](const vec3i_t& idx) { return nodes[getCellIdx(idx)]; }
  node_t& operator[](uint i) { return nodes[i]; }
};

#endif /* REGGRID_H_ */
