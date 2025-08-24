/*
 * texture.cpp
 *
 *  Created on: Nov 9, 2010
 *      Author: denis
 */

#include "texture.h"

#include <GL/glew.h>

#include "image_helper.h"

namespace smoldyn {

Texture::Texture() : id(0), tmuBound(-1) {}

Texture::~Texture() {}

bool Texture::Load(const std::string& name) {
  return load_png(name.c_str(), bitmap);
}

void Texture::Upload() {
  glGenTextures(1, (GLuint*)&id);
  glBindTexture(GL_TEXTURE_2D, id);

  glTexImage2D(GL_TEXTURE_2D, 0, bitmap.bpp, bitmap.width, bitmap.height, 0,
               (bitmap.bpp == 4) ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE,
               bitmap.pixels);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR);  // Linear Filtering
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                  GL_LINEAR);  // Linear Filtering
}

void Texture::Bind(int tmu) { glBindTexture(GL_TEXTURE_2D, id); }

void Texture::Unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

void Texture::Unload() { glDeleteTextures(1, (GLuint*)&id); }
/*
private:
        int		id;
        int		tmuBound;
        bitmap_t	bitmap;
};
*/
}  // namespace smoldyn
