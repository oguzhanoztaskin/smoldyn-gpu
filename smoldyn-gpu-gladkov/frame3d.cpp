/*
 * 3dframe.cpp
 *
 *  Created on: Feb 18, 2011
 *      Author: denis
 */

#include "frame3d.h"

Frame3D::Frame3D() : dirty(true) {}

Frame3D::~Frame3D() {}

void Frame3D::SetPosition(const vec3_t& v) {
  position = v;
  dirty = true;
}

void Frame3D::SetRotation(const vec3_t& v) {
  rotation = v;
  dirty = true;
}

void Frame3D::SetScale(const vec3_t& v) {
  scale = v;
  dirty = true;
}

vec3_t Frame3D::GetPosition() const { return position; }

vec3_t Frame3D::GetRotation() const { return rotation; }

vec3_t Frame3D::GetScale() const { return scale; }

matrix4x4 Frame3D::GetMatrix() {
  if (dirty) {
    CalculateMatrix();
    dirty = false;
  }

  return matrix;
}

void Frame3D::CalculateMatrix() {
  matrix4x4 t = matrix4x4::translation(position);
  matrix4x4 r = matrix4x4::rotation(rotation);
  matrix4x4 s = matrix4x4::scale(scale);

  matrix = t * r * s;
}

/*
        vec3_t	position;
        vec3_t	rotation;
        vec3_t	scale;

        matrix4x4	matrix;

        bool	dirty;
*/
