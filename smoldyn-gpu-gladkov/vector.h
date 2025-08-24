/*
 * vector.h
 *
 *  Created on: Mar 15, 2010
 *      Author: denis
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "config.h"

struct vec3_t {
  float x, y, z;

  vec3_t(float _x = 0, float _y = 0, float _z = 0) { reset(_x, _y, _z); }

  void reset(float _x = 0, float _y = 0, float _z = 0) {
    x = _x;
    y = _y;
    z = _z;
  }

  vec3_t& operator+=(const vec3_t& c) {
    x += c.x;
    y += c.y;
    z += c.z;

    return *this;
  }

  vec3_t& operator-=(const vec3_t& c) {
    x -= c.x;
    y -= c.y;
    z -= c.z;

    return *this;
  }

  vec3_t& operator*=(float v) {
    x *= v;
    y *= v;
    z *= v;

    return *this;
  }

  vec3_t& operator/=(float v) {
    x /= v;
    y /= v;
    z /= v;

    return *this;
  }

  vec3_t& normalize();
  float length() const;
  vec3_t& negate();

  static vec3_t neg(const vec3_t& a);
  static float dot(const vec3_t& a, const vec3_t& b);
  static vec3_t norm(const vec3_t& a);
  static vec3_t cross(const vec3_t& a, const vec3_t& b);
};

vec3_t calc_normal(const vec3_t& v1, const vec3_t& v2, const vec3_t& v3);

vec3_t operator+(const vec3_t& a, const vec3_t& b);
vec3_t operator-(const vec3_t& a, const vec3_t& b);
vec3_t operator/(const vec3_t& a, float v);
vec3_t operator*(const vec3_t& a, float v);

struct vec4_t {
  float x, y, z, w;

  explicit vec4_t(float _x = 0, float _y = 0, float _z = 0, float _w = 0) {
    reset(_x, _y, _z, _w);
  }

  explicit vec4_t(const vec3_t& v) { reset(v.x, v.y, v.z, 1.0f); }

  void reset(float _x = 0, float _y = 0, float _z = 0, float _w = 0) {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }

  vec4_t& operator+=(const vec4_t& c) {
    x += c.x;
    y += c.y;
    z += c.z;
    w += c.w;

    return *this;
  }

  vec4_t& operator-=(const vec4_t& c) {
    x -= c.x;
    y -= c.y;
    z -= c.z;
    w -= c.w;

    return *this;
  }

  vec4_t& operator*=(float v) {
    x *= v;
    y *= v;
    z *= v;
    w *= v;

    return *this;
  }

  vec4_t& operator/=(float v) {
    x /= v;
    y /= v;
    z /= v;
    w /= v;

    return *this;
  }

  vec3_t to_vec3() { return vec3_t(x, y, z); }

  static vec4_t from_vec3(const vec3_t& v) {
    return vec4_t(v.x, v.y, v.z, 1.0f);
  }
};

struct vec3i_t {
  uint x, y, z;

  vec3i_t(uint _x = 0, uint _y = 0, uint _z = 0) { reset(_x, _y, _z); }

  void reset(uint _x = 0, uint _y = 0, uint _z = 0) {
    x = _x;
    y = _y;
    z = _z;
  }
};

struct vec2_t {
  float x, y;

  vec2_t() { reset(); }

  void reset(float _x = 0, float _y = 0) {
    x = _x;
    y = _y;
  }
};

#endif /* VECTOR_H_ */
