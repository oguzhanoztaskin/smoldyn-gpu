/*
 * cmd_handler.h
 *
 *  Created on: Dec 15, 2010
 *      Author: denis
 */

#ifndef CMD_HANDLER_H_
#define CMD_HANDLER_H_

#include <sstream>

namespace smolgpu {

class smolparams_t;

class SmoldynSolver;

class cmd_handler {
 public:
  virtual void process(SmoldynSolver& s) = 0;
  virtual void dump(std::ostream& out, const smolparams_t& p) const = 0;
};

cmd_handler* CreateCmd(std::istringstream& ss, smolparams_t& p);

}  // namespace smolgpu

#endif /* CMD_HANDLER_H_ */
