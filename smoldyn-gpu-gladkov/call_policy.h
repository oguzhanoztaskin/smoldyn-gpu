/*
 * call_policy.h
 *
 *  Created on: Dec 15, 2010
 *      Author: denis
 */

#ifndef CALL_POLICY_H_
#define CALL_POLICY_H_

#include <string>

#include "log_file.h"

#include <sstream>

namespace	smolgpu
{

class	smolparams_t;

class	call_policy
{
public:
	virtual bool	canCall(float time, int iteration) = 0;
	virtual void	dump(std::ostream& ) {}
};

call_policy*	CreateCallPolicy(std::istringstream& ss, const smolparams_t& p);

}

#endif /* CALL_POLICY_H_ */
