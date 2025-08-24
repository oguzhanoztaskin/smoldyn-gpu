/*
 * gpu_dsmc.h
 *
 *  Created on: Jun 1, 2010
 *      Author: denis
 */

#ifndef GPU_DSMC_H_
#define GPU_DSMC_H_

#include "dsmc_base.h"
#include "particles.h"

#include "color_map.h"

namespace	dsmc
{

	template	<class	DataStrategy>
	class	GPUDSMCSolver: public	DSMCSolver
	{
	public:
		GPUDSMCSolver();
		~GPUDSMCSolver();

	protected:

		void	RunSimulationStep(float dt);
		void	RunBenchmarkSimulationStep(float dt);

		void	Render();
		void	InitData();

		statistics_t*	SampleStatistics(float4& collStat);

		uint	InitSystemState(const float3& streamVel);

		void	ReorderParticles(float4* pos, float4* vel, float3* col,
								float4*	old_pos, float4* old_vel, float3* old_col);

		void	UpdateSortedGrid(float4* pos);

		void	InitRegularGrid(reg_grid_t& grid, const float* v, uint size);

		void	RenderConcMap(uint	x, uint	y, uint	w, uint	h);

		unsigned char*	GetColorMap(uint& w, uint& h, uint&	bpp);

		float3*	GetVelocityField();

		DataStrategy	particles;

		ColorMap<gpu_data_provider_t>	colorMap;

		float*	concentration;
		uint*	gridHashes;
		uint*	cellGridIndices;
		uint2*	cellStartEnd;
		uint*	geomHashes;
	};
}

#include "gpu_dsmc.inl"

#endif /* GPU_DSMC_H_ */
