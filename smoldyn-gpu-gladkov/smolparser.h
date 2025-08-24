/*
 * smolparser.h
 *
 *  Created on: Oct 20, 2010
 *      Author: denis
 */

#ifndef SMOLPARSER_H_
#define SMOLPARSER_H_

#include "smolparameters.h"
namespace	smolgpu
{

class	SmoldynParser
{
public:
	SmoldynParser();
	~SmoldynParser();
	void	operator()(const char*	filename, smolparams_t& p);
};

}

#endif /* SMOLPARSER_H_ */
