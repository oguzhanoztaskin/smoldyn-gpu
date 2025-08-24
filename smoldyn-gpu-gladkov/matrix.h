/*
 * matrix.h
 *
 *  Created on: Apr 9, 2010
 *      Author: denis
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "vector.h"

struct matrix4x4 {
  float m[4][4];

  float& operator()(uint i, uint j) { return m[i][j]; }

  matrix4x4 operator*(const matrix4x4& m);
  vec3_t operator*(const vec3_t& v);
  vec4_t operator*(const vec4_t& v);

  matrix4x4& inverse();
  matrix4x4& transpose();
  matrix4x4& invtrans();

  static matrix4x4 inverse(const matrix4x4& m);
  static matrix4x4 transpose(const matrix4x4& m);
  static matrix4x4 invtrans(const matrix4x4& m);
  static matrix4x4 identity();

  static matrix4x4 translation(const vec3_t& v);
  static matrix4x4 rotation(const vec3_t& v);
  static matrix4x4 scale(const vec3_t& v);
};

#endif /* MATRIX_H_ */
