/*
 * gas_props.h
 *
 *  Created on: Jun 1, 2010
 *      Author: denis
 */

#ifndef GAS_PROPS_H_
#define GAS_PROPS_H_

#include "dsmc.h"

namespace	dsmc
{
	struct	gas_props_t
	{
		float	molmass;
		float	diameter;
		float	Tref;

		gas_props_t(): molmass(DSMC_MOL_MASS), diameter(DSMC_DIAM), Tref(DSMC_T0) {}
	};
}

#endif /* GAS_PROPS_H_ */
