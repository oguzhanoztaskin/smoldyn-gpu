/*
 * geom_kernels.cu
 *
 *  Created on: Feb 21, 2011
 *      Author: denis
 */

#include "geom_kernels.cuh"
#include "tri_intersect.cu"
#include "vectors.cu"

namespace smolgpu {

geom_node_t::~geom_node_t() {
  if (geom.devIndices) cudaFree(geom.devIndices);

  if (geom.devIndicesStartEnd) cudaFree(geom.devIndicesStartEnd);

  if (geom.devVertices) cudaFree(geom.devVertices);

  if (geom.devVerticesWorld) cudaFree(geom.devVerticesWorld);

  if (geom.devFaceNormals) cudaFree(geom.devFaceNormals);

  if (geom.devHashes) cudaFree(geom.devHashes);
}

void geom_node_t::transform(Frame3D& f) {
  moved = true;

  matrix4x4 m = f.GetMatrix();

  matrix[0] = make_float4(m.m[0][0], m.m[0][1], m.m[0][2], m.m[0][3]);
  matrix[1] = make_float4(m.m[1][0], m.m[1][1], m.m[1][2], m.m[1][3]);
  matrix[2] = make_float4(m.m[2][0], m.m[2][1], m.m[2][2], m.m[2][3]);
  matrix[3] = make_float4(m.m[3][0], m.m[3][1], m.m[3][2], m.m[3][3]);
}

namespace kernels {

#define COLLISION_LIMIT 10
#define EPS 0.0001f

#define MODEL_WIDTH 0.0001f

__host__ void UploadGeometryHashes(geom_node_t& node, reg_grid_t& grid) {
  uint* host_has_geom = (uint*)malloc(grid.cellCount() * sizeof(uint));

  for (uint i = 0; i < grid.cellCount(); i++)
    host_has_geom[i] = grid[i].second ? 1 : 0;

  cudaMalloc(&node.geom.devHashes, grid.cellCount() * sizeof(uint));
  cudaMemcpy(node.geom.devHashes, host_has_geom,
             grid.cellCount() * sizeof(uint), cudaMemcpyHostToDevice);

  free(host_has_geom);
}

void UploadIndices(geom_node_t& node, reg_grid_t& grid) {
  uint idxCountTotal = 0;
  for (uint i = 0; i < grid.cellCount(); i++)
    idxCountTotal += grid[i].first.size();

  uint3* host_indices = (uint3*)malloc(idxCountTotal * sizeof(uint));
  uint2* host_cellStEnd = (uint2*)malloc(grid.cellCount() * sizeof(uint2));

  float3* host_normals = (float3*)malloc(idxCountTotal / 3 * sizeof(float3));

  memset(host_cellStEnd, 0xff, grid.cellCount() * sizeof(uint2));

  uint offset = 0;
  for (uint i = 0; i < grid.cellCount(); i++) {
    uint idxCount = grid[i].first.size();

    if (idxCount) {
      reg_grid_t::idx_vector_t& n = grid[i].first;
      host_cellStEnd[i].x = offset;
      for (uint k = 0; k < idxCount; k += 3) {
        host_indices[offset + k / 3].x = n[k + 0];
        host_indices[offset + k / 3].y = n[k + 1];
        host_indices[offset + k / 3].z = n[k + 2];

        vec3_t& norm = grid.normals[i][k / 3];

        host_normals[offset + k / 3].x = norm.x;
        host_normals[offset + k / 3].y = norm.y;
        host_normals[offset + k / 3].z = norm.z;
      }
      offset += idxCount / 3;
      host_cellStEnd[i].y = offset;
    }
  }

  cudaMalloc(&node.geom.devIndices, idxCountTotal * sizeof(uint));
  cudaMemcpy(node.geom.devIndices, host_indices, idxCountTotal * sizeof(uint),
             cudaMemcpyHostToDevice);

  cudaMalloc(&node.geom.devIndicesStartEnd, grid.cellCount() * sizeof(uint2));
  cudaMemcpy(node.geom.devIndicesStartEnd, host_cellStEnd,
             grid.cellCount() * sizeof(uint2), cudaMemcpyHostToDevice);

  cudaMalloc(&node.geom.devFaceNormals, idxCountTotal / 3 * sizeof(float3));
  cudaMemcpy(node.geom.devFaceNormals, host_normals,
             idxCountTotal / 3 * sizeof(float3), cudaMemcpyHostToDevice);

  node.geom.facesSize = idxCountTotal / 3;

  free(host_indices);
  free(host_cellStEnd);
  free(host_normals);
}
/*
 * 	struct	collision_grid_t
        {
                float3	mmin;
                float3	mmax;
                float3	cellSize;
                uint3	dim;
        };
 */

float3 v3tof3(const vec3_t& v) { return make_float3(v.x, v.y, v.z); }

uint3 v3tof3(const vec3i_t& v) { return make_uint3(v.x, v.y, v.z); }

__host__ void CreateGeometryNodeDev(geom_node_t& node, Model& model) {
  node.transform(model.GetFrame());

  node.grid.cellSize = v3tof3(model.GetGrid().cellSize);
  node.grid.dim = v3tof3(model.GetGrid().dim);
  node.grid.mmax = v3tof3(model.GetGrid().gridSize.max);
  node.grid.mmin = v3tof3(model.GetGrid().gridSize.min);

  node.type = model.GetType();
  node.prob = model.GetProbablility();

  UploadGeometryHashes(node, model.GetGrid());
  UploadIndices(node, model.GetGrid());

  const float* vertices = &model.GetSurfaces()[0].vertices[0].x;
  const uint vertCount = model.GetSurfaces()[0].vertices.size();

  node.geom.vertSize = vertCount;

  cudaMalloc(&node.geom.devVertices, vertCount * sizeof(float3));
  cudaMalloc(&node.geom.devVerticesWorld, vertCount * sizeof(float3));
  cudaMemcpy(node.geom.devVertices, vertices, vertCount * sizeof(float3),
             cudaMemcpyHostToDevice);
}

__device__ uint GetCollisionCellID(const collision_grid_t& grid, float3 pos) {
  if (pos.x < grid.mmin.x || pos.y < grid.mmin.y || pos.z < grid.mmin.z ||

      pos.x > grid.mmax.x || pos.y > grid.mmax.y || pos.z > grid.mmax.z)

    return 0xffffffff;

  uint x = (pos.x - grid.mmin.x) / grid.cellSize.x;
  uint y = (pos.y - grid.mmin.y) / grid.cellSize.y;
  uint z = (pos.z - grid.mmin.z) / grid.cellSize.z;

  x = min(x, grid.dim.x - 1);
  y = min(y, grid.dim.y - 1);
  z = min(z, grid.dim.z - 1);

  return z * grid.dim.x * grid.dim.y + y * grid.dim.x + x;
}

__device__ bool IsGeometryInCell(float3 cmin, uint* geom_hashes,
                                 const collision_grid_t& grid) {
  float3 p = cmin;
  uint cid = 0xffffffff;
  float3 csize = grid.cellSize;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }

  p.x += csize.x;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }

  p.y += csize.y;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }

  p.z += csize.z;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }

  p = cmin;

  p.y += csize.y;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }

  p.z += csize.z;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }

  p.y = cmin.y;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }
  p.x += csize.x;

  if ((cid = GetCollisionCellID(grid, p)) != 0xffffffff) {
    if (geom_hashes[cid] != 0) return true;
  }
  return false;
}

__global__ void TransformGeometryDev(float3* vert_local, float3* vert_global,
                                     uint vertSize, float4 m0, float4 m1,
                                     float4 m2, float4 m3) {
  const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= vertSize) return;

  float3 v = vert_local[idx], r;

  r.x = v.x * m0.x + v.y * m0.y + v.z * m0.z + m0.w;
  r.y = v.x * m1.x + v.y * m1.y + v.z * m1.z + m1.w;
  r.z = v.x * m2.x + v.y * m2.y + v.z * m2.z + m2.w;

  vert_global[idx] = r;
}

__device__ bool CollideWithGeometryDev(molecule_t& currm, molecule_t& prevm,
                                       const uint3* indices,
                                       const uint2* indicesStartEnd,
                                       const float3* vertices,
                                       const float3* normals, uint cellid,
                                       const int type, const float prob,
                                       cudahelp::rand::Xor128Generator& gen) {
  uint2 idx_stend = indicesStartEnd[cellid];

  float3 v1, v2, v3, p, norm;

  float3 dir = normalize(make_float3(currm.pos.x - prevm.pos.x,
                                     currm.pos.y - prevm.pos.y,
                                     currm.pos.z - prevm.pos.z));
  float t, u, v;

  float dist = distance(prevm.pos, currm.pos);

  if (dist < EPS) return false;

  float3 width;

  for (uint i = idx_stend.x; i < idx_stend.y; i++) {
    uint3 idx = indices[i];

    v1 = vertices[idx.x];
    v2 = vertices[idx.y];
    v3 = vertices[idx.z];

    if (intersect_triangle(prevm.pos, dir, v1, v2, v3, &t, &u, &v)) {
      if (t >= 0 && t <= dist)  // intersection
      {
        bool doReflect = true;

        if (type == AbsorbProb) {
          if (gen.GetFloat() > prob) doReflect = false;
        }

        if (type == Reflect || (type == AbsorbProb && doReflect)) {
          float uv = 1 - u - v;

          norm = normals[i];

          width = make_float3(norm.x * MODEL_WIDTH, norm.y * MODEL_WIDTH,
                              norm.z * MODEL_WIDTH);

          p.x = v1.x * uv + v2.x * u + v3.x * v;
          p.y = v1.y * uv + v2.y * u + v3.y * v;
          p.z = v1.z * uv + v2.z * u + v3.z * v;

          float3 p1p = reflect(dir, norm);

          float od = dist - t;

          currm.pos.x = p.x + p1p.x * od;
          currm.pos.y = p.y + p1p.y * od;
          currm.pos.z = p.z + p1p.z * od;

          prevm.pos.x = p.x + width.x;
          prevm.pos.y = p.y + width.y;
          prevm.pos.z = p.z + width.z;

          return true;
        }

        currm.type = -1;
        return true;
      }
    }
  }

  return false;
}
/*
 * molecule_t& currm, molecule_t& prevm, const uint3* indices,
                                                                                                const uint2* indicesStartEnd, const float3*	vertices, const float3* normals, uint cellid)
 */

__global__ void ProcessCollReflectiveDev(collision_grid_t grid,
                                         geom_desc_t geom, molecule_t* curr,
                                         molecule_t* prev, uint numParticles,
                                         const uint type, const float prob) {
  const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= numParticles) return;

  uint st_cubid = 0xffffffff, end_cubid = 0xffffffff;

  bool collProcessed = false;

  collProcessed = false;

  molecule_t currm = curr[idx];
  molecule_t prevm = prev[idx];

  if (currm.type == -1) return;

  uint i = 0;

  cudahelp::rand::Xor128Generator gen(idx);

  do {
    st_cubid = GetCollisionCellID(grid, prevm.pos);
    end_cubid = GetCollisionCellID(grid, currm.pos);

    if (st_cubid != 0xffffffff) {
      if (geom.devHashes[st_cubid] != 0)
        collProcessed = CollideWithGeometryDev(
            currm, prevm, geom.devIndices, geom.devIndicesStartEnd,
            geom.devVerticesWorld, geom.devFaceNormals, st_cubid, type, prob,
            gen);
    }

    if (!collProcessed && st_cubid != end_cubid && end_cubid != 0xffffffff &&
        currm.type != -1) {
      if (geom.devHashes[end_cubid] != 0)
        collProcessed = CollideWithGeometryDev(
            currm, prevm, geom.devIndices, geom.devIndicesStartEnd,
            geom.devVerticesWorld, geom.devFaceNormals, end_cubid, type, prob,
            gen);
    }

  } while (collProcessed && i++ < COLLISION_LIMIT && currm.type != -1);

  curr[idx] = currm;
}

__host__ void ProcessCollisions(geom_node_t& node, molecule_t* curr,
                                molecule_t* prev, int size) {
  if (node.moved) TransformGeometry(node);

  if (node.type == Transparent) return;

  uint numThreads = 128;
  uint numBlocks = GetNumberOfBlocks(numThreads, size);

  ProcessCollReflectiveDev<<<numBlocks, numThreads>>>(
      node.grid, node.geom, curr, prev, size, node.type, node.prob);
}

__host__ void TransformGeometry(geom_node_t& node) {
  uint numThreads = 128;
  uint numBlocks = GetNumberOfBlocks(numThreads, node.geom.vertSize);

  TransformGeometryDev<<<numBlocks, numThreads>>>(
      node.geom.devVertices, node.geom.devVerticesWorld, node.geom.vertSize,
      node.matrix[0], node.matrix[1], node.matrix[2], node.matrix[3]);

  node.moved = false;
}

}  // namespace kernels
}  // namespace smolgpu
