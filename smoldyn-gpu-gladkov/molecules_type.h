/*
 * molecules_type.h
 *
 *  Created on: Nov 3, 2010
 *      Author: denis
 */

#ifndef MOLECULES_TYPE_H_
#define MOLECULES_TYPE_H_

namespace	smolgpu
{
	const	int	DeadParticleId = -1;

	struct	molecule_t
	{
		float3	pos;
		int		type;
	};
}

#endif /* MOLECULES_TYPE_H_ */
