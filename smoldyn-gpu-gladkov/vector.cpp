/*
 * vector.cpp
 *
 *  Created on: Mar 15, 2010
 *      Author: denis
 */

#include "vector.h"

#include "math.h"

vec3_t vec3_t::cross(const vec3_t& a, const vec3_t& b) {
  vec3_t c;

  c.x = a.y * b.z - a.z * b.y;
  c.y = a.z * b.x - a.x * b.z;
  c.z = a.x * b.y - a.y * b.x;

  return c;
}

float vec3_t::dot(const vec3_t& a, const vec3_t& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3_t& vec3_t::negate() {
  x *= -1;
  y *= -1;
  z *= -1;

  return *this;
}

vec3_t vec3_t::neg(const vec3_t& a) {
  vec3_t b = a;
  return b.negate();
}

vec3_t& vec3_t::normalize() {
  float l = length();

  if (l != 0) {
    x /= l;
    y /= l;
    z /= l;
  }

  return *this;
}

vec3_t calc_normal(const vec3_t& v1, const vec3_t& v2, const vec3_t& v3) {
  return vec3_t::norm(vec3_t::cross(v2 - v1, v3 - v1));
}

vec3_t operator+(const vec3_t& a, const vec3_t& b) {
  vec3_t c = a;

  return c += b;
}

vec3_t operator-(const vec3_t& a, const vec3_t& b) {
  vec3_t c = a;

  return c -= b;
}

vec3_t operator/(const vec3_t& a, float v) {
  vec3_t t = a;
  return t /= v;
}

vec3_t operator*(const vec3_t& a, float v) {
  vec3_t t = a;
  return t *= v;
}

float vec3_t::length() const { return sqrtf(x * x + y * y + z * z); }

vec3_t vec3_t::norm(const vec3_t& a) {
  vec3_t b = a;
  b.normalize();
  return b;
}
