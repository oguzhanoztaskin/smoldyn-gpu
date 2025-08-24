/*
 *  model.h
 *
 *
 *  Created by Denis Gladkov on 1/28/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>

#include "aabb.h"
#include "config.h"
#include "frame3d.h"
#include "reggrid.h"
#include "vector.h"

/*
 * store vertices, texcoords, normals etc in separate arrays for all subsets
 * flat store 2 sets of indices:
 * 1. subsurface indices
 * 2. grid structure
 */

// #define	MODEL_NO_GRAPHICS

enum SurfaceType { Reflect = 0, Absorb = 1, AbsorbProb = 2, Transparent = 3 };

class Model {
 public:
  Model();
  ~Model();

  struct surface_t;

  bool Load(const char*, bool calcNormals);
  // TODO: use collada???
  bool LoadRaw(const char*, bool createBuffers = true);
  bool SaveRaw(const char*);

  const std::vector<Model::surface_t>& GetSurfaces() { return surfaces; }

  bbox_t GetBBox() { return bbox; }

  void SetName(const std::string& s) { name = s; }

  const std::string& GetName() const { return name; }

  void SetPosition(const vec3_t& v) { frame.SetPosition(v); }
  void SetRotation(const vec3_t& v) { frame.SetRotation(v); }
  void SetScale(const vec3_t& v) { frame.SetScale(v); }

  void SetType(uint t) { type = t; }
  uint GetType() { return type; }

  float GetProbablility() { return prob; }
  void SetProbablility(float f) { prob = f; }

  Frame3D& GetFrame() { return frame; }

  void SetFrame(const Frame3D& f) { frame = f; }

  void Render(bool normals);
  void RenderGrid();
  void Reset();

  void RenderTestGrid();
  void RenderTestGrid(uint);
  void RenderTestGrid(const vec3i_t& idx);

  void TestAABB(const AABB_t& aabb);

  reg_grid_t& GetGrid() { return grid; }

  void MoveModel(const vec3_t& translate);

  void CreateGrid(const vec3i_t& dim);

  void CreateBuffers();

  struct surface_t {
    unsigned int vertBuffer;
    unsigned int indexBuffer;
    unsigned int texCoordBuffer;
    unsigned int normalsBuffer;
    unsigned int colorsBuffer;

    bbox_t bbox;

    std::vector<uint> indices;
    std::vector<vec3_t> vertices;
    std::vector<vec3_t> colors;
    std::vector<vec3_t> normals;
    std::vector<vec3_t> facesNorm;
    std::vector<vec2_t> texCoords;

    std::string name;

    surface_t();
    ~surface_t();

    void Render();
    void RenderNormals();
    void Reset();
    void CalcNormals();
    void CreateBuffers();
    void CalcBBox();

    void CalcFacesNormals();

    static Model::surface_t merge(const std::vector<surface_t>& surfaces,
                                  const std::string& name);
    static bbox_t merge_bbox(const std::vector<surface_t>& surfaces);
  };

 private:
  reg_grid_t grid;
  std::vector<surface_t> surfaces;
  bbox_t bbox;

  std::string name;

  int type;
  float prob;

  Frame3D frame;
};

#endif
