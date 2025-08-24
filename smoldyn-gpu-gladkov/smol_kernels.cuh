/*
 * smol_kernels.cuh
 *
 *  Created on: Nov 1, 2010
 *      Author: denis
 */

#ifndef SMOL_KERNELS_CUH_
#define SMOL_KERNELS_CUH_

#include "smolparameters.h"

#include "cudahelp/random/mtwister.h"

#include	"molecules_type.h"

namespace	smolgpu
{

struct	molcount_stat_t
{
	int	id;
	int	count;

	molcount_stat_t(): id(0), count(0)
	{

	}
};

struct	diff_stat_t: public molcount_stat_t
{
	float3	avgPos;
	float3	m1; //xy, xz, yz//shear
	float3	m2; //x2, y2, z2//normal

	diff_stat_t()
	{

	}
};

namespace	kernels
{

__host__	void	InitSimulationParameters(const smolparams_t&	params);
__host__	void	InitSpecies(molecule_t*	mols, float3* colors, uint*	isAlive, const smolparams_t&	sp);
__host__	uint	CalculateDiffusion(molecule_t*	mols, float3* colors, uint molsAlive);
__host__	uint	CalcAliveSpecies(molecule_t*	mols);
__host__	void	SortSpecies(molecule_t*	mols, float3*	colors);
__host__	void	HashMolecules(molecule_t*	mols, uint size);

__host__	void	HashMoleculesAtomics(molecule_t*	mols, uint size);

__host__	uint	DoZerothReactions(molecule_t*	mols, float3*	colors);
__host__	uint	DoFirstReactions(molecule_t*	mols, float3*	colors, uint size);
__host__	uint	DoSecondReactions(molecule_t*	mols, float3*	colors, uint size);

__host__	uint	DoSecondReactionsNewHash(molecule_t*	mols, float3*	colors, uint size);

__host__	uint	RemoveDeadMolecules(molecule_t*	mols, float3* color, uint size);

__host__	void	CalculateDiffStatistics(std::vector<diff_stat_t>& stat, molecule_t* mols, int actSize);
__host__	void	CalculateMolcountStatistics(std::vector<molcount_stat_t>& stat, molecule_t* mols, int actSize);
__host__	void	CalculateMolCountSpace(std::vector<int>& v, molecule_t* mols, int actSize, int axis, int bins, int specId, float3 bmin, float3 bmax);
__host__	void	ProcessEquimol(molecule_t* mols, float3* colors, int size, int spec1, int spec2, float prob);
__host__	void	DeinitSpecies();

}

}

#endif /* SMOL_KERNELS_CUH_ */
