/*
 * matrix.cpp
 *
 *  Created on: Apr 9, 2010
 *      Author: denis
 */

#include "matrix.h"

#include <math.h>

#include "memory.h"

matrix4x4 matrix4x4::operator*(const matrix4x4& m1) {
  matrix4x4 mm;

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float c = 0.0f;
      for (int k = 0; k < 4; k++) c += m[i][k] * m1.m[k][j];

      mm.m[i][j] = c;
    }

  return mm;
}

vec3_t matrix4x4::operator*(const vec3_t& v) {
  vec4_t v4(v);

  return ((*this) * v4).to_vec3();
}

vec4_t matrix4x4::operator*(const vec4_t& v) {
  vec4_t v4;

  v4.x = v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2] + v.w * m[0][3];
  v4.y = v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2] + v.w * m[1][3];
  v4.z = v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2] + v.w * m[2][3];
  v4.w = v.x * m[3][0] + v.y * m[3][1] + v.z * m[3][2] + v.w * m[3][3];

  return v4;
}

matrix4x4& matrix4x4::inverse() { return *this; }

matrix4x4& matrix4x4::transpose() {
  *this = transpose(*this);
  return *this;
}

matrix4x4& matrix4x4::invtrans() { return *this; }

matrix4x4 matrix4x4::inverse(const matrix4x4& m) {
  matrix4x4 mm = m;
  return mm.inverse();
}

matrix4x4 matrix4x4::transpose(const matrix4x4& m) {
  matrix4x4 mm;

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) mm.m[i][j] = m.m[j][i];

  return mm;
}

matrix4x4 matrix4x4::invtrans(const matrix4x4& m) {
  matrix4x4 mm = m;
  return mm.invtrans();
}

matrix4x4 matrix4x4::identity() {
  matrix4x4 m;
  memset(&m.m[0][0], 0, 16 * sizeof(float));

  m.m[0][0] = m.m[1][1] = m.m[2][2] = m.m[3][3] = 1.0f;

  return m;
}

matrix4x4 matrix4x4::translation(const vec3_t& v) {
  matrix4x4 m = identity();

  m.m[0][3] = v.x;
  m.m[1][3] = v.y;
  m.m[2][3] = v.z;

  return m;
}

matrix4x4 matrix4x4::rotation(const vec3_t& v) {
  matrix4x4 m = identity();

  float cx = cosf(v.x);
  float sx = sinf(v.x);

  float cy = cosf(v.y);
  float sy = sinf(v.y);

  float cz = cosf(v.z);
  float sz = sinf(v.z);

  m.m[0][0] = cz;
  m.m[1][0] = sx * sy * cz + cx * sz;
  m.m[2][0] = -cx * sy * cz + sx * sz;

  m.m[0][1] = -cy * sz;
  m.m[1][1] = -sx * sy * sz + cx * cz;
  m.m[2][1] = cx * sy * sz + sx * cz;

  m.m[0][2] = sy;
  m.m[1][2] = -sx * cy;
  m.m[2][2] = cx * cy;

  return m;
}

matrix4x4 matrix4x4::scale(const vec3_t& v) {
  matrix4x4 m = identity();

  m.m[0][0] = v.x;
  m.m[1][1] = v.y;
  m.m[2][2] = v.z;

  return m;
}
