/*
 * smol_cmd.h
 *
 *  Created on: Dec 15, 2010
 *      Author: denis
 */

#ifndef SMOL_CMD_H_
#define SMOL_CMD_H_

#include "call_policy.h"
#include "cmd_handler.h"

namespace smolgpu {

class SmoldynSolver;

class SmoldynCmd {
 public:
  SmoldynCmd(cmd_handler* ch, call_policy* cp = 0) : handler(ch), callp(cp) {}
  void process(float time, int iteration, SmoldynSolver& solver) {
    if (!handler) return;

    if (callp) {
      if (callp->canCall(time, iteration)) handler->process(solver);
    } else
      handler->process(solver);
  }

  void dump(std::ostream& out, const smolparams_t& p) const {
    if (callp) callp->dump(out);
    if (handler) handler->dump(out, p);
  }

 private:
  call_policy* callp;
  cmd_handler* handler;
};
}  // namespace smolgpu

#endif /* SMOL_CMD_H_ */
