/*
 * color_map.h
 *
 *  Created on: Jun 1, 2010
 *      Author: denis
 */

#ifndef COLOR_MAP_H_
#define COLOR_MAP_H_

#include "config.h"

namespace dsmc {
// create either on gpu or cpu
// the same to cpu and gpu implementation
// characterized by strategy

struct null_data_provider_t {
  void lock_data(uint width, uint height) {}
  void unlock_data() {}
  void init(uint w, uint h) {}
  void deinit() {}
};

struct gpu_data_provider_t {
  void init(uint w, uint h);
  void lock_data(uint width, uint height);
  void unlock_data();
  void deinit();

  uint colorBuffer;
};

template <class DataProvider>
class ColorMap {
 public:
  void Deinit();
  void Render(uint x, uint y, uint w, uint h);
  void Init(uint width, uint height);
  DataProvider& GetDataProvider() { return m_dataProvider; }

 private:
  DataProvider m_dataProvider;
  uint texId;
  uint m_width;
  uint m_height;
};
}  // namespace dsmc

#include "color_map.inl"

#endif /* COLOR_MAP_H_ */
