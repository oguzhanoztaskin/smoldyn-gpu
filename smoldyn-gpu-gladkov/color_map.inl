/*
 * color_map.cpp
 *
 *  Created on: Jun 3, 2010
 *      Author: denis
 */

#include <GL/glew.h>

#include "draw_helper.h"

namespace dsmc {

template <class DataProvider>
void ColorMap<DataProvider>::Render(uint x, uint y, uint w, uint h) {
  glBindTexture(GL_TEXTURE_2D, texId);

  glColor3f(1, 1, 1);

  m_dataProvider.lock_data(m_width, m_height);

  glBegin(GL_QUADS);

  glTexCoord2f(0, 0);
  glVertex2i(x, y);

  glTexCoord2f(0, 1);
  glVertex2i(x, y + h);

  glTexCoord2f(1, 1);
  glVertex2i(x + w, y + h);

  glTexCoord2f(1, 0);
  glVertex2i(x + w, y);

  glEnd();

  m_dataProvider.unlock_data();

  glBindTexture(GL_TEXTURE_2D, 0);
}

template <class DataProvider>
void ColorMap<DataProvider>::Deinit() {
  m_dataProvider.deinit();
}

template <class DataProvider>
void ColorMap<DataProvider>::Init(uint width, uint height) {
  m_width = width;
  m_height = height;

  texId = CreateTexture(m_width, m_height);
  m_dataProvider.init(m_width, m_height);
}

/*
                DataProvider	m_dataProvider;
                uint	texId;
*/
}  // namespace dsmc
