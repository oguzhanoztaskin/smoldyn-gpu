/*
 * smol_solver.h
 *
 *  Created on: Nov 1, 2010
 *      Author: denis
 */

#ifndef SMOL_SOLVER_H_
#define SMOL_SOLVER_H_

#include	"smolparameters.h"
//struct	diff_stat_t: public molcount_stat_t

#include "smol_kernels.cuh"

namespace	smolgpu
{

class	SmoldynSolver
{
public:
	SmoldynSolver();
	~SmoldynSolver();

	void	Init(const	smolparams_t& params);
	void	Update(float	time);
	void	Render();
	void	RenderModels();

	const std::vector<diff_stat_t>&	GetMolMoments();
	const std::vector<molcount_stat_t>& GetMolCount();
	const std::vector<int>& GetMolCountSpace(int axis, int bins, int specId, float3 bmin, float3 bmax);

	void	ProcessEquimol(int spec1, int spec2, float prob);

	void	ToggleRenderGrid();
	void	ToggleRenderNormals();

	float	GetSimulationTime();

private:
	struct	impl_t;
	impl_t*	m_pImpl;
};

}

#endif /* SMOL_SOLVER_H_ */
