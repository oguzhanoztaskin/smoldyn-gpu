/*
 * 3dframe.h
 *
 *  Created on: Feb 18, 2011
 *      Author: denis
 */

#ifndef _3DFRAME_H_
#define _3DFRAME_H_

#include "matrix.h"
#include "vector.h"

class Frame3D {
 public:
  Frame3D();
  ~Frame3D();

  void SetPosition(const vec3_t& v);
  void SetRotation(const vec3_t& v);
  void SetScale(const vec3_t& v);

  vec3_t GetPosition() const;
  vec3_t GetRotation() const;
  vec3_t GetScale() const;

  matrix4x4 GetMatrix();

 private:
  void CalculateMatrix();

  vec3_t position;
  vec3_t rotation;
  vec3_t scale;

  matrix4x4 matrix;

  bool dirty;
};

#endif /* 3DFRAME_H_ */
