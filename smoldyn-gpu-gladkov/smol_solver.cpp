/*
 * smol_solver.cpp
 *
 *  Created on: Nov 1, 2010
 *      Author: denis
 */

#include "smol_solver.h"

#include "smol_kernels.cuh"

#include "cudahelp/random/rndfast.h"

#include "molecules.h"

#include <iostream>

#include <fstream>

#include "log_file.h"

#include "draw_helper.h"

#include "geom_kernels.cuh"

#include <cuda_runtime.h>

namespace	smolgpu
{

struct	gnodes_cmp
{
	bool	operator()(smolgpu::geom_node_ptr a, smolgpu::geom_node_ptr b)
	{
		return a->type < b->type;
	}
};


struct	SmoldynSolver::impl_t
{
	smolparams_t	m_params;
	smolgpu::molecule_state_vbo_t	molecules;
	std::vector<smolgpu::diff_stat_t>	diffStat;
	std::vector<smolgpu::molcount_stat_t>	countStat;
	std::vector<int>	countSpaceStat;

	typedef	std::vector<smolgpu::geom_node_ptr>	geom_nodes_t;

	geom_nodes_t	geom_nodes;

	SmoldynSolver*	upper;

	int	iteration;

	bool	bRenderGrid;
	bool	bRenderNormals;

	impl_t(SmoldynSolver* u):iteration(0), upper(u), bRenderGrid(false), bRenderNormals(false)
	{
	}

	~impl_t()
	{
		kernels::DeinitSpecies();
	}

	void	init(const	smolparams_t&	p)
	{
		m_params = p;

		diffStat.resize(p.species.size());

		kernels::InitSimulationParameters(p);

		cudahelp::rand::InitFastRngs(p.max_mol);

		molecules.init(m_params.max_mol);

		molecule_t*	mols = molecules.lock_molecules();

		kernels::InitSpecies(mols, molecules.lock_colors(), molecules.isAlive, m_params);

		molecules.copy_prev_mols(mols);

		molecules.unlock_colors();

		molecules.aliveMolecules = kernels::CalcAliveSpecies(mols);//molecules.isAlive);

		molecules.unlock_molecules();

		LogFile::get()<<"Generated "<<molecules.aliveMolecules<<" molecules\n";

		smolparams_t::models_map_t::iterator	it = m_params.models.begin(), end = m_params.models.end();

		for(; it != end; ++it)
		{
			Frame3D& f = it->second.GetFrame();
			it->second.GetGrid().transform(f);
			it->second.CreateBuffers();

			geom_node_ptr	n(new geom_node_t);

			kernels::CreateGeometryNodeDev(*n, it->second);

			geom_nodes.push_back(n);
		}

		gnodes_cmp	c;

		std::sort(geom_nodes.begin(), geom_nodes.end(), c);
	}

	float	GetSimTime()
	{
		return iteration*m_params.timeStep+m_params.startTime;
	}

	const std::vector<diff_stat_t>&	GetMolMoments()
	{
		return diffStat;
	}

	const std::vector<int>& GetMolCountSpace(int axis, int bins, int specId, float3 bmin, float3 bmax)
	{
		molecule_t*	mols = molecules.lock_molecules();

		countSpaceStat.resize(bins);

		kernels::CalculateMolCountSpace(countSpaceStat, mols, molecules.aliveMolecules, axis, bins, specId, bmin, bmax);

		molecules.unlock_molecules();

		return countSpaceStat;
	}

	void	ProcessEquimol(int spec1, int spec2, float prob)
	{
		molecule_t*	mols = molecules.lock_molecules();
		float3* cols = molecules.lock_colors();

		kernels::ProcessEquimol(mols, cols, molecules.aliveMolecules, spec1, spec2, prob);

		molecules.unlock_colors();
		molecules.unlock_molecules();
	}

	const std::vector<molcount_stat_t>& GetMolCount()
	{
		return countStat;
	}

/*
 * time ctr  v1[0] v1[1] v1[2] m1[0] m1[1] m1[2] m1[3] m1[4] m1[5] m1[6] m1[7] m1[8]
 0   1000   0     0     0     0     0     0     0     0     0     0     0     0
 */
	void	outputStatistics(const diff_stat_t&	s, const std::string& str)
	{
		std::ofstream	ostr(str.c_str(), std::ios_base::app);

		ostr<<iteration*m_params.timeStep+m_params.startTime<<" "<<s.count<<" "<<s.avgPos.x<<" "
				<<s.avgPos.y<<" "<<s.avgPos.z<<" "<<s.m2.x<<" "<<s.m1.x<<" "<<s.m1.y<<" "<<s.m1.x
				<<" "<<s.m2.y<<" "<<s.m1.z<<" "<<s.m1.y<<" "<<s.m1.z<<" "<<s.m2.z<<"\n";
	}

	void	update(float f)
	{
		molecule_t*	mols = molecules.lock_molecules();
		float3* col = molecules.lock_colors();

#if 1

		if(m_params.dumpMolMoments)
			kernels::CalculateDiffStatistics(diffStat, mols,molecules.aliveMolecules);

		if(m_params.dumpMolCounts)
			kernels::CalculateMolcountStatistics(countStat, mols,molecules.aliveMolecules);
#endif

		molecules.aliveMolecules = kernels::CalculateDiffusion(mols, col, molecules.aliveMolecules);

		bool	removeDeadPart = false;

		for(int i = 0; i < geom_nodes.size(); i++)
		{
			if(geom_nodes[i]->type == Absorb || geom_nodes[i]->type == AbsorbProb)
				removeDeadPart = true;

			kernels::ProcessCollisions(*geom_nodes[i], mols, molecules.prev_mols, molecules.aliveMolecules);
		}

		if(removeDeadPart)
			molecules.aliveMolecules = kernels::RemoveDeadMolecules(mols, col, molecules.aliveMolecules);

		molecules.copy_prev_mols(mols);

		if(m_params.zeroreact.size() > 0)
			molecules.aliveMolecules = kernels::DoZerothReactions(mols, col);

		if(m_params.firstreact.size() > 0)
			molecules.aliveMolecules = kernels::DoFirstReactions(mols, col, molecules.aliveMolecules);

		if(m_params.secondreact.size() > 0)
		{
			kernels::HashMolecules(mols, molecules.aliveMolecules);
			molecules.aliveMolecules = kernels::DoSecondReactions(mols, col, molecules.aliveMolecules);

//			kernels::HashMoleculesAtomics(mols, molecules.aliveMolecules);
//			molecules.aliveMolecules = kernels::DoSecondReactionsNewHash(mols, col, molecules.aliveMolecules);
		}

		molecules.unlock_colors();
		molecules.unlock_molecules();

		for(int i = 0; i < m_params.cmds.size(); i++)
			m_params.cmds[i].process(GetSimTime(), iteration, *upper);

		iteration++;
	}

	void	render_models()
	{
		smolparams_t::models_map_t::iterator	it = m_params.models.begin(), end = m_params.models.end();

		for(; it != end; ++it)
		{
			RenderModel(it->second, bRenderNormals);

			if(bRenderGrid)
				it->second.RenderGrid();
		}
	}

	void	render()
	{
		molecules.render();
	}
};

SmoldynSolver::SmoldynSolver(): m_pImpl(new impl_t(this))
{

}

SmoldynSolver::~SmoldynSolver()
{
	delete m_pImpl;
}

void	SmoldynSolver::ToggleRenderGrid()
{
	m_pImpl->bRenderGrid = !m_pImpl->bRenderGrid;
}

void	SmoldynSolver::ToggleRenderNormals()
{
	m_pImpl->bRenderNormals = !m_pImpl->bRenderNormals;
}

float	SmoldynSolver::GetSimulationTime()
{
	return m_pImpl->GetSimTime();
}

const std::vector<diff_stat_t>&	SmoldynSolver::GetMolMoments()
{
	return m_pImpl->GetMolMoments();
}

const std::vector<molcount_stat_t>& SmoldynSolver::GetMolCount()
{
	return m_pImpl->GetMolCount();
}

const std::vector<int>& SmoldynSolver::GetMolCountSpace(int axis, int bins, int specId, float3 bmin, float3 bmax)
{
	return m_pImpl->GetMolCountSpace(axis, bins, specId, bmin, bmax);
}

void	SmoldynSolver::ProcessEquimol(int spec1, int spec2, float prob)
{
	return m_pImpl->ProcessEquimol(spec1, spec2, prob);
}

void	SmoldynSolver::Init(const	smolparams_t& params)
{
	m_pImpl->init(params);
}

void	SmoldynSolver::Update(float	time)
{
	m_pImpl->update(time);
}

void	SmoldynSolver::Render()
{
	m_pImpl->render();
}

void	SmoldynSolver::RenderModels()
{
	m_pImpl->render_models();
}

}
