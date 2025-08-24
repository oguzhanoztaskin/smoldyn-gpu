/*
 * texture.h
 *
 *  Created on: Nov 9, 2010
 *      Author: denis
 */

#ifndef TEXTURE_H_
#define TEXTURE_H_

#include <string>

#include "bitmap.h"

namespace smoldyn {
class Texture {
 public:
  Texture();
  ~Texture();

  bool Load(const std::string& name);
  void Upload();
  void Bind(int tmu);
  void Unbind();
  void Unload();

 private:
  int id;
  int tmuBound;
  bitmap_t bitmap;
};
}  // namespace smoldyn

#endif /* TEXTURE_H_ */
