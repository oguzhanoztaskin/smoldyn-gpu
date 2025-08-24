/*
 * log_file.cpp
 *
 *  Created on: Dec 15, 2010
 *      Author: denis
 */

#include "log_file.h"

#include <stdexcept>

namespace smolgpu {

LogFile* LogFile::m_instance = 0;

void LogFile::create(std::ostream& o) {
  if (m_instance != 0) delete m_instance;

  m_instance = new LogFile(o);
}

LogFile& LogFile::get() {
  if (!m_instance) throw std::runtime_error("Uninitialized LogFile");

  return *m_instance;
}

void LogFile::destroy() {
  if (m_instance) {
    m_instance->get_stream().flush();
    delete m_instance;
  }
}

}  // namespace smolgpu
