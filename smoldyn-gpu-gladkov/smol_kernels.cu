/*
 * smol_kernels.cu
 *
 *  Created on: Nov 1, 2010
 *      Author: denis
 */

#include	"smol_kernels.cuh"

#include	"config.h"

#include	"cudahelp/random/mtwister.cuh"
#include	"cudahelp/random/rndfast.cuh"

#include	"thrust/count.h"
#include	"thrust/fill.h"
#include	"thrust/sort.h"

#include 	"thrust/remove.h"

#include	"bound_cond.cuh"

#include 	"gpu_mutex.cuh"

#include 	"part_hash.cuh"

#define		SPECIES_NUMBER	10

#define		STATES_COUNT	5

#include "radixsort.h"

nvRadixSort::RadixSort*	radixSort = 0;

uint	g_sortBits = 0;

struct	smol_gpu_params_t
{
	float3	bmin;
	float3	bmax;

	uint3	dim;
	float3	csize;

	float	timestep;
};

struct	gpu_grid_t
{
	uint2*	cellStartEnd;
	uint* 	indices;
	uint* 	hashes;
	uint*	offsets;

	uint	gridDim;

	gpu_grid_t(): cellStartEnd(0), indices(0), hashes(0), gridDim(0), offsets(0) {}
};

gpu_grid_t	devGrid;

__device__	__constant__	smol_gpu_params_t	g_parameters;

float*	devDiffStep = 0;
float*	devDiffC = 0;
float*	devGaussTable = 0;

int*	devNearest = 0;

uint*	devHistogram = 0;
uint	g_histSize = 0;

struct	momentum_t
{
	float3	m1;
	float3	m2;
};

molecule_t*	devTmpMolecules = 0;
momentum_t*	devMomentums = 0;

int	g_gaussTableSize = 0;

uint*	g_freeIdx = 0;

smolgpu::smolparams_t	g_cpuParams;

//generates molecules of the kind Type
//is supposed to be running concurrently for different molecules types
template	<class	DistX, class	DistY, class	DistZ>
__global__	void	InitSpeciesDev(smolgpu::molecule_t*	mol, float3* colors, uint* isAlive, float3 col, int count, int type, DistX distx, DistY disty, DistZ distz)
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

	smolgpu::molecule_t	m;

	colors[idx] = col;

	float3	mn = g_parameters.bmin;
	float3	mx = g_parameters.bmax;

	m.pos.x = distx(idx, mn.x, mx.x-mn.x);
	m.pos.y = disty(idx, mn.y, mx.y-mn.y);
	m.pos.z = distz(idx, mn.z, mx.z-mn.z);

	m.type = type;

	mol[idx] = m;
}

__global__	void	CalculateDiffusionDev(smolgpu::molecule_t*	mol, float* diffstep, float*	gtable, int tsize, int count)
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

	smolgpu::molecule_t	m = mol[idx];

	if(m.type == -1)
		return;

	float	step = diffstep[m.type*STATES_COUNT];

	cudahelp::rand::Xor128Generator gen(idx);

	m.pos.x += step*gtable[gen.GetInt(tsize)];
	m.pos.y += step*gtable[gen.GetInt(tsize)];
	m.pos.z += step*gtable[gen.GetInt(tsize)];

	mol[idx] = m;
}

__global__	void	UpdateDiffStepsDev(float*	diffsteps, float*	diffc, int	size, float	dt)
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;

	if(idx >= size)
		return;

	diffsteps[idx] = sqrtf(2.0f*diffc[idx]*dt);
}

struct	DistributeConstant
{
	DistributeConstant(float	p): pos(p) {}
	__device__	float	operator()(int	id, float l, float size) {return pos;}

	float	pos;
};

struct	DistributeUniformFast
{
	__device__	float	operator()(int	id, float l, float size)
	{
		cudahelp::rand::Xor128Generator gen(id);
		return l + gen.GetFloat()*size;
	}
};

struct	DistributeUniformTwister
{
	__device__	float	operator()(int	id, float l, float size)
	{
		cudahelp::rand::MTGenerator	gen(id);
		return l + gen.GetFloat()*size;
	}
};

__device__	inline uint3	GetCellID3(float3 pos)
{
	uint x = (pos.x - g_parameters.bmin.x) / g_parameters.csize.x;
	uint y = (pos.y - g_parameters.bmin.y) / g_parameters.csize.y;
	uint z = (pos.z - g_parameters.bmin.z) / g_parameters.csize.z;

	return make_uint3(min(x,g_parameters.dim.x-1), min(y,g_parameters.dim.y-1), min(z,g_parameters.dim.z-1));
}

__device__	inline uint	MakeCellID3(uint3 idx)
{
	return idx.z*g_parameters.dim.x*g_parameters.dim.y + idx.y*g_parameters.dim.x + idx.x;
}

__device__	unsigned int	GetCellID(float3 pos)
{
	uint x = (pos.x - g_parameters.bmin.x) / g_parameters.csize.x;
	uint y = (pos.y - g_parameters.bmin.y) / g_parameters.csize.y;
	uint z = (pos.z - g_parameters.bmin.z) / g_parameters.csize.z;

	return min(z,g_parameters.dim.z-1)*g_parameters.dim.x*g_parameters.dim.y +
			min(y,g_parameters.dim.y-1)*g_parameters.dim.x +
				min(x,g_parameters.dim.x-1);
}

__device__	float3	GetCell(float3 pos)
{
	uint x = (pos.x - g_parameters.bmin.x) / g_parameters.csize.x;
	uint y = (pos.y - g_parameters.bmin.y) / g_parameters.csize.y;
	uint z = (pos.z - g_parameters.bmin.z) / g_parameters.csize.z;

	x = min(x,g_parameters.dim.x-1);
	y = min(y,g_parameters.dim.y-1);
	z = min(z,g_parameters.dim.z-1);

	return make_float3(g_parameters.bmin.x + x*g_parameters.csize.x,
						g_parameters.bmin.y + y*g_parameters.csize.y,
						g_parameters.bmin.z + z*g_parameters.csize.z);
}

namespace	smolgpu
{

void	GenerateGaussTable(float*, int);
int poisrandD(double xm);

namespace	kernels
{

uint	RemoveDeadMolecules(molecule_t*	mols, float3* color, uint size);

__host__	void	RunInitSpeciesKernel(const	species_t& sp, float3* colorss, uint*	isAlive, float3 color, molecule_t* mols)
{
	molecule_t* mol = mols;
	float3*	colors = colorss;

	for(int i = 0; i < sp.distr.size(); i++)
	{
		int	d = sp.distr[i].bool3ToInt();

		int	count = sp.distr[i].count;

		float3	pos = sp.distr[i].pos;

		int	numThreads = 128;
		int	numBlocks  = GetNumberOfBlocks(numThreads, count);

		switch(d)
		{
		case 7:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeUniformFast(), DistributeUniformFast(), DistributeUniformFast());
			break;
		case 6:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeUniformFast(), DistributeUniformFast(), DistributeConstant(pos.z));
			break;
		case 5:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeUniformFast(), DistributeConstant(pos.y),DistributeUniformFast());
			break;
		case 4:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeUniformFast(), DistributeConstant(pos.y),DistributeConstant(pos.z));
			break;
		case 3:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeConstant(pos.x),DistributeUniformFast(), DistributeUniformFast());
			break;
		case 2:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeConstant(pos.x),DistributeUniformFast(), DistributeConstant(pos.z));
			break;
		case 1:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeConstant(pos.x),DistributeConstant(pos.y), DistributeUniformFast());
			break;
		case 0:
			InitSpeciesDev<<<numBlocks,numThreads>>>(mol, colors, isAlive, color, count, sp.id, DistributeConstant(pos.x),DistributeConstant(pos.y), DistributeConstant(pos.z));
			break;
		}

		mol += count;
		colors += count;
	}
}

__host__	void	InitSpecies(molecule_t*	mols, float3* colors, uint*	isAlive, const smolparams_t&	sp)
{
	thrust::device_ptr<molecule_t> dev_data(mols);
	thrust::device_ptr<molecule_t> dev_data_end(mols + g_cpuParams.max_mol);

	molecule_t	deadMol = {make_float3(0,0,0), -1};

	thrust::fill(dev_data, dev_data_end, deadMol);

	smolparams_t::species_map_t::const_iterator it = sp.species.begin();
	smolparams_t::species_map_t::const_iterator end = sp.species.end();

	molecule_t*	mol = mols;
	float3*	cc = colors;
	uint*	alive = isAlive;

	//TODO: parallel kernel execution
	for(; it != end; ++it)
	{
		RunInitSpeciesKernel(it->second, cc, alive, it->second.color, mol);
		mol += it->second.getNumberOfMols();//count;
		cc += it->second.getNumberOfMols();
		alive += it->second.getNumberOfMols();
	}

	int	numThreads = 128;

	int	statesCount = g_cpuParams.species.size()*STATES_COUNT;

	int numBlocks  = GetNumberOfBlocks(numThreads, statesCount);

	UpdateDiffStepsDev<<<numBlocks, numThreads>>>(devDiffStep, devDiffC, statesCount, g_cpuParams.timeStep);
}

__host__	uint	CalcAliveSpecies(uint*	isAlive)
{
	thrust::device_ptr<uint> dev_data(isAlive);
	thrust::device_ptr<uint> dev_data_end(isAlive + g_cpuParams.max_mol);

	return	thrust::count(dev_data, dev_data_end, 1);
}

struct is_alive_pred
{
	__host__ __device__
	bool operator()(const smolgpu::molecule_t& m)
	{
		return m.type != -1;
	}
};

struct mol_type_pred
{
	int	type;

	__host__ __device__
	mol_type_pred(int tp): type(tp) {}

	__host__ __device__
	bool operator()(const smolgpu::molecule_t& m)
	{
		return m.type == type;
	}
};

struct mols_greater
{
	__host__ __device__
	bool operator()(const smolgpu::molecule_t& m1, const smolgpu::molecule_t& m2)
	{
		return m1.type > m2.type;
	}
};

__host__	void	SortSpecies(molecule_t*	mols, float3*	colors)
{
	thrust::device_ptr<molecule_t> dev_data(mols);
	thrust::device_ptr<molecule_t> dev_data_end(mols + g_cpuParams.max_mol);

	thrust::device_ptr<float3> colors_data(colors);

	thrust::sort_by_key(dev_data, dev_data_end, colors_data, mols_greater());
}

__host__	uint	CalcAliveSpecies(molecule_t*	mols)
{
	thrust::device_ptr<molecule_t> dev_data(mols);
	thrust::device_ptr<molecule_t> dev_data_end(mols + g_cpuParams.max_mol);

	return	thrust::count_if(dev_data, dev_data_end, is_alive_pred());
}

__host__	void	RunZeroReactionsKernel(molecule_t*	mols, float3*	col, int	type,	int size,	float3	color)
{
	int	numThreads = 128;
	int	numBlocks  = GetNumberOfBlocks(numThreads, size);

	InitSpeciesDev<<<numBlocks,numThreads>>>(mols, col, 0, color, size, type, DistributeUniformFast(), DistributeUniformFast(), DistributeUniformFast());

//	printf("Generating %d molecules of type %d\n", size, type);
}

__host__	uint	DoZerothReactions(molecule_t*	mols, float3*	colors)
{
	smolparams_t::zeroreact_map_t::const_iterator it = g_cpuParams.zeroreact.begin();
	smolparams_t::zeroreact_map_t::const_iterator end = g_cpuParams.zeroreact.end();

	std::vector<uint>	species(g_cpuParams.species.size(),0);

	for(;it != end; ++it)
	{
		uint	nMol = poisrandD(it->second.prob);

//		printf("Generating %d molecules of type %s. Prob: %f\n", nMol, it->second.name.c_str(),it->second.prob);

		for(uint i = 0; i < it->second.products.size(); i++)
			species[it->second.products[i]] += nMol;
	}

	uint	aliveSpec = CalcAliveSpecies(mols);

	uint	emptySpots = g_cpuParams.max_mol - aliveSpec;

	molecule_t*	m = mols + aliveSpec;
	float3*	c = colors + aliveSpec;

	for(uint i = 0; i < species.size() && emptySpots > 0; i++)
	{
		if(species[i] != 0)
		{
			uint	size = min(species[i], emptySpots);

			smolgpu::species_t	s = g_cpuParams.getSpiece(i);

			RunZeroReactionsKernel(m, c, i, size, s.color);

			m += size;
			c += size;

			emptySpots -= size;
			aliveSpec += size;
		}
	}

	return aliveSpec;
}

__global__	void	FirstReactionsKernelDev(molecule_t*	mol, float3*	col, int	intype, int outtype, int p2, int count, uint* freeIdx, float tresh, float3	color, float3 c2)
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

	smolgpu::molecule_t	m = mol[idx];

	if(m.type != intype)
		return;

	cudahelp::rand::Xor128Generator gen(idx);

	if(gen.GetFloat() < tresh)
	{
		float3	cs = g_parameters.csize;
		float3	cell = GetCell(m.pos);

//		if(outtype != -1)
//		{
			m.pos.x = cell.x + gen.GetFloat()*cs.x;
			m.pos.y = cell.y + gen.GetFloat()*cs.y;
			m.pos.z = cell.z + gen.GetFloat()*cs.z;

			col[idx] = color;

			m.type = outtype;

			mol[idx] = m;
//		}
//		else
//		{
//			int	last = atomicAdd(freeIdx, -1) - 1;
//
//			mol[idx] = mol[last];
//			col[idx] = col[last];
//			mol[last].type = -1;
//		}



		if(p2 != -1)
		{
			m.pos.x = cell.x + gen.GetFloat()*cs.x;
			m.pos.y = cell.y + gen.GetFloat()*cs.y;
			m.pos.z = cell.z + gen.GetFloat()*cs.z;

			idx = atomicAdd(freeIdx, 1);

			col[idx] = c2;
			m.type = p2;

			mol[idx] = m;
		}
	}
}

__host__	void	RunFirstReactionsKernel(molecule_t*	mols, float3*	col, int size, const firstreact_t& rc)
{
	int	numThreads = 128;
	int	numBlocks  = GetNumberOfBlocks(numThreads, size);

	float3	c = make_float3(1,1,1);
	float3	c2 = c;

	if(rc.products[0] != -1)
		c = g_cpuParams.getSpiece(rc.products[0]).color;

	int	prod2 = -1;

	if(rc.products.size() > 1)
	{
		prod2 = rc.products[1];
		c2 = g_cpuParams.getSpiece(prod2).color;
	}

	FirstReactionsKernelDev<<<numBlocks,numThreads>>>(mols, col, rc.reactant, rc.products[0], prod2, size, g_freeIdx, rc.prob, c, c2);
}

__host__	uint	DoFirstReactions(molecule_t*	mols, float3*	colors, uint size)
{
	cudaMemcpy(g_freeIdx, &size, sizeof(uint), cudaMemcpyHostToDevice);

	smolparams_t::firstreact_map_t::const_iterator it = g_cpuParams.firstreact.begin();
	smolparams_t::firstreact_map_t::const_iterator end = g_cpuParams.firstreact.end();

	bool	doCompaction = false;

	uint	newSize = size;

	for(;it != end; ++it)
	{
		if(it->second.deallocMolecules())
			doCompaction = true;

		RunFirstReactionsKernel(mols, colors, newSize, it->second);

		cudaMemcpy(&newSize, g_freeIdx, sizeof(uint), cudaMemcpyDeviceToHost);
	}


	if(doCompaction)
		newSize = RemoveDeadMolecules(mols, colors, newSize);

	return newSize;
}

__device__	float	distance2(float3 v1, float3 v2)
{
	return (v2.x-v1.x)*(v2.x-v1.x) + (v2.y-v1.y)*(v2.y-v1.y) + (v2.z-v1.z)*(v2.z-v1.z);
}

__device__	int		FindNearestMolecule(molecule_t*	mol, molecule_t& m, int react2,
											float bindrad2, uint*	indices, uint2* cells, uint cellid,
											float&	min_dst, int& min_idx)
{
	uint2	cellStEnd = cells[cellid];
	int		partCount = cellStEnd.y - cellStEnd.x;

	if(partCount < 2)
		return min_idx;

	for(int i = cellStEnd.x; i < cellStEnd.y; i++)
	{
		int	idx2 = indices[i];

		molecule_t	m2 = mol[idx2];

		if(m2.type == -1 || m2.type != react2)
			continue;

		float dst = distance2(m.pos, m2.pos);

		if(dst <= bindrad2 && dst < min_dst)
		{
			min_dst = dst;
			min_idx = idx2;

			break;
		}
	}

	return min_idx;
}

__device__	int		FindNearestMoleculeNewHash(molecule_t*	mol, molecule_t& m, int react2,
											float bindrad2, uint*	indices, uint* offsets, uint cellid,
											float&	min_dst, int& min_idx)
{
	uint	cellStart = offsets[cellid];
	int		partCount = offsets[cellid+1] - cellStart;

	if(partCount < 2)
		return min_idx;

	for(int i = cellStart; i < partCount; i++)
	{
		int	idx2 = indices[i];

		molecule_t	m2 = mol[idx2];

		if(m2.type == -1 || m2.type != react2)
			continue;

		float dst = distance2(m.pos, m2.pos);

		if(dst <= bindrad2 && dst < min_dst)
		{
			min_dst = dst;
			min_idx = idx2;
		}
	}

	return min_idx;
}
/*
__device__	bool	ProcessSecondOrderInCell2(molecule_t*	mol, molecule_t& m, float3*	col, int react2, uint idx,
												int p1, int p2, float bindrad2, float3	color, float3 c2,
												uint*	indices, uint2* cells, uint3 cell_idx, cudahelp::rand::Xor128Generator& gen)
{
	float	min_dst = 100000000;
	int	min_idx = -1;

#if 1

	uint cellid = MakeCellID3(cell_idx);

	min_idx = FindNearestMolecule(mol, m, react2, bindrad2, indices, cells, cellid, min_dst, min_idx);
#else

	for(int x = max(cell_idx.x-1,0); x <= min(cell_idx.x+1,g_parameters.dim.x-1); x++)
		for(int y = max(cell_idx.y-1,0); y <= min(cell_idx.y+1,g_parameters.dim.y-1); y++)
			for(int z = max(cell_idx.z-1,0); z <= min(cell_idx.z+1,g_parameters.dim.z-1); z++)
			{
				uint	cellid = GetCellID3(make_uint3(x,y,z));
				min_idx = FindNearestMolecule(mol, m, react2, bindrad2, indices, cells, cellid, min_dst, min_idx);
			}
#endif

	if(min_idx == -1)
		return false;

	cudahelp::GPUMutex	mutex2(min_idx, idx);

//	if(!mutex2.Locked())
//		return false;

	molecule_t	m2 = mol[min_idx];

//	if(m2.type != react2)
//		return false;

	float3	cs = g_parameters.csize;
	float3	cell = GetCell(m.pos);

	if(p1 != -1)
	{
		m.pos.x = cell.x + gen.GetFloat()*cs.x;
		m.pos.y = cell.y + gen.GetFloat()*cs.y;
		m.pos.z = cell.z + gen.GetFloat()*cs.z;

		col[idx] = color;
	}

	m.type = p1;

	mol[idx] = m;

	if(p2 != -1)
	{
		m2.pos.x = cell.x + gen.GetFloat()*cs.x;
		m2.pos.y = cell.y + gen.GetFloat()*cs.y;
		m2.pos.z = cell.z + gen.GetFloat()*cs.z;

		col[min_idx] = c2;
	}

	m2.type = p2;

	mol[min_idx] = m2;

	return true;
}
*/
__global__	void	FindNearestMoleculesDev(molecule_t*	mol, int	r1, int r2, int count, float bindrad2,
												uint*	indices, uint2* cells, int*	nearest, const bool useNeighbors)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

	molecule_t	m = mol[idx];

	int	react2 = r2;

	int	min_idx = -1;

	if(m.type == r1 && m.type != -1)// || m.type == r2)
	{
//		if(m.type == r2)
//			react2 = r1;

		uint3 cell_idx = GetCellID3(m.pos);

		float	min_dst = 100000000;

		if(!useNeighbors)
		{
			uint cellid = MakeCellID3(cell_idx);

			min_idx = FindNearestMolecule(mol, m, react2, bindrad2, indices, cells, cellid, min_dst, min_idx);
		}
		else
		{
			for(int x = max(cell_idx.x-1,0); x <= min(cell_idx.x+1,g_parameters.dim.x-1); x++)
				for(int y = max(cell_idx.y-1,0); y <= min(cell_idx.y+1,g_parameters.dim.y-1); y++)
					for(int z = max(cell_idx.z-1,0); z <= min(cell_idx.z+1,g_parameters.dim.z-1); z++)
					{
						uint	cellid = MakeCellID3(make_uint3(x,y,z));
						min_idx = FindNearestMolecule(mol, m, react2, bindrad2, indices, cells, cellid, min_dst, min_idx);

						if(min_idx != -1)
							break;
					}
		}
	}

	nearest[idx] = min_idx;
}

__global__	void	FindNearestMoleculesNewHashDev(molecule_t*	mol, int	r1, int r2, int count, float bindrad2,
												uint*	indices, uint* offsets, int*	nearest, const bool useNeighbors)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

	molecule_t	m = mol[idx];

	int	react2 = r2;

	int	min_idx = -1;

	if(m.type == r1 || m.type == r2)
	{
		if(m.type == r2)
			react2 = r1;

		uint3 cell_idx = GetCellID3(m.pos);

		float	min_dst = 100000000;

		if(!useNeighbors)
		{
			uint cellid = MakeCellID3(cell_idx);

			min_idx = FindNearestMoleculeNewHash(mol, m, react2, bindrad2, indices, offsets, cellid, min_dst, min_idx);
		}
		else
		{
			for(int x = max(cell_idx.x-1,0); x <= min(cell_idx.x+1,g_parameters.dim.x-1); x++)
				for(int y = max(cell_idx.y-1,0); y <= min(cell_idx.y+1,g_parameters.dim.y-1); y++)
					for(int z = max(cell_idx.z-1,0); z <= min(cell_idx.z+1,g_parameters.dim.z-1); z++)
					{
						uint	cellid = MakeCellID3(make_uint3(x,y,z));
						min_idx = FindNearestMoleculeNewHash(mol, m, react2, bindrad2, indices, offsets, cellid, min_dst, min_idx);
					}
		}
	}

	nearest[idx] = min_idx;
}

__global__	void	SecondReactionsKernelDev2(molecule_t*	mol, float3*	col,
											int p1, int p2, int count, float3	color,
											float3 c2, int*	nearest)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

#ifdef	USE_MUTEX
	cudahelp::GPUMutex	mutex(idx, idx);

	if(!mutex.Locked())
		return;
#endif

	molecule_t	m = mol[idx];

	if(m.type == -1)
		return;

	cudahelp::rand::Xor128Generator gen(idx);

	int	idx2 = nearest[idx];

	if(idx2 == -1 || idx2 >= count)
		return;

	molecule_t	m2 = mol[idx2];

#ifdef	USE_MUTEX
	cudahelp::GPUMutex	mutex2(idx2, idx);

	if(!mutex2.Locked())
		return;
#endif

	float3	cs = g_parameters.csize;
	float3	cell = GetCell(m.pos);

	if(p1 != -1)
	{
		m.pos.x = cell.x + gen.GetFloat()*cs.x;
		m.pos.y = cell.y + gen.GetFloat()*cs.y;
		m.pos.z = cell.z + gen.GetFloat()*cs.z;

		col[idx] = color;
	}

	m.type = p1;

	mol[idx] = m;

	if(p2 != -1)
	{
		m2.pos.x = cell.x + gen.GetFloat()*cs.x;
		m2.pos.y = cell.y + gen.GetFloat()*cs.y;
		m2.pos.z = cell.z + gen.GetFloat()*cs.z;

		col[idx2] = c2;
	}

	m2.type = p2;

	mol[idx2] = m2;
}
/*
__global__	void	SecondReactionsKernelNewDev(molecule_t*	mol, float3*	col, int	r1, int r2,
											int p1, int p2, int count, uint* freeIdx, float bindrad2,
											float3	color, float3 c2,
											uint*	indices, uint2* cells)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= count)
		return;

	cudahelp::GPUMutex	mutex(idx, idx);

	if(!mutex.Locked())
		return;

	molecule_t	m = mol[idx];

	int	react2 = r2;

	if(m.type == r1 || m.type == r2)
	{
		if(m.type == r2)
			react2 = r1;

		cudahelp::rand::Xor128Generator gen(idx);

		uint3 cell_idx = GetCellID3(m.pos);

		ProcessSecondOrderInCell2(mol, m, col, react2, idx, p1, p2, bindrad2, color, c2, indices, cells, cell_idx, gen);
	}
}
*/
__global__	void	SecondReactionsKernelDev(molecule_t*	mol, float3*	col, int	r1, int r2,
											int p1, int p2, int count, uint* freeIdx, float bindrad2,
											float3	color, float3 c2,
											uint*	indices, uint2* cells)
{
	const	uint	cellid = blockDim.x*gridDim.x*blockIdx.y + blockIdx.x*blockDim.x + threadIdx.x;

	uint2	cellStEnd = cells[cellid];
	int		partCount = cellStEnd.y - cellStEnd.x;

	if(partCount < 2)
		return;

	cudahelp::rand::Xor128Generator gen(cellid);

	for(int i = cellStEnd.x; i < cellStEnd.y; i++)
	{
		int	idx = indices[i];

		molecule_t	m = mol[idx];

		if(m.type == -1)
			continue;

		int	react2 = r2;

		if(m.type == r1 || m.type == r2)
		{
			if(m.type == r2)
				react2 = r1;

			for(int j = i+1; j < cellStEnd.y; j++)
			{
				int	idx2 = indices[j];
				molecule_t	m1 = mol[idx2];

				if(m1.type == react2)
				{
					if(distance2(m.pos, m1.pos) <= bindrad2)
					{
						float3	cs = g_parameters.csize;
						float3	cell = GetCell(m.pos);

						if(p1 != -1)
						{
							m.pos.x = cell.x + gen.GetFloat()*cs.x;
							m.pos.y = cell.y + gen.GetFloat()*cs.y;
							m.pos.z = cell.z + gen.GetFloat()*cs.z;

							col[idx] = color;
						}

						m.type = p1;

						mol[idx] = m;

						if(p2 != -1)
						{
							m1.pos.x = cell.x + gen.GetFloat()*cs.x;
							m1.pos.y = cell.y + gen.GetFloat()*cs.y;
							m1.pos.z = cell.z + gen.GetFloat()*cs.z;

							col[idx2] = c2;
						}

						m1.type = p2;

						mol[idx2] = m1;

						break;
					}
				}
			}
		}
	}

}

//One thread per cell so far
void	RunSecondReactionsKernel(molecule_t*	mols, float3*	col, int size, const secondreact_t& rc)
{
	float3	c = make_float3(1,1,1);
	float3	c2 = c;

	if(rc.products[0] != -1)
		c = g_cpuParams.getSpiece(rc.products[0]).color;

	int	prod2 = -1;

	if(rc.products.size() > 1)
	{
		prod2 = rc.products[1];
		c2 = g_cpuParams.getSpiece(prod2).color;
	}

	dim3 dimGrid;
	dimGrid.x = devGrid.gridDim;
	dimGrid.y = devGrid.gridDim;
	dimGrid.z = 1;

#if 0
	SecondReactionsKernelDev<<<dimGrid, devGrid.gridDim>>>(mols, col, rc.reactant, rc.reactant2, rc.products[0], prod2,
			size, g_freeIdx, rc.bindrad2, c, c2, devGrid.indices, devGrid.cellStartEnd);
#else
	uint	numThreads = 512;
	uint	numBlocks = GetNumberOfBlocks(numThreads, size);

	FindNearestMoleculesDev<<<numBlocks,numThreads>>>(mols, rc.reactant, rc.reactant2, size, rc.bindrad2,
								devGrid.indices, devGrid.cellStartEnd, devNearest, g_cpuParams.accuracy > 5);

	SecondReactionsKernelDev2<<<numBlocks,numThreads>>>(mols, col, rc.products[0], prod2,  size, c, c2, devNearest);

#endif
}

void	RunSecondReactionsKernelNewHash(molecule_t*	mols, float3*	col, int size, const secondreact_t& rc)
{
	float3	c = make_float3(1,1,1);
	float3	c2 = c;

	if(rc.products[0] != -1)
		c = g_cpuParams.getSpiece(rc.products[0]).color;

	int	prod2 = -1;

	if(rc.products.size() > 1)
	{
		prod2 = rc.products[1];
		c2 = g_cpuParams.getSpiece(prod2).color;
	}

	dim3 dimGrid;
	dimGrid.x = devGrid.gridDim;
	dimGrid.y = devGrid.gridDim;
	dimGrid.z = 1;

#if 0
	SecondReactionsKernelDev<<<dimGrid, devGrid.gridDim>>>(mols, col, rc.reactant, rc.reactant2, rc.products[0], prod2,
			size, g_freeIdx, rc.bindrad2, c, c2, devGrid.indices, devGrid.cellStartEnd);
#else
	uint	numThreads = 512;
	uint	numBlocks = GetNumberOfBlocks(numThreads, size);

	FindNearestMoleculesNewHashDev<<<numBlocks,numThreads>>>(mols, rc.reactant, rc.reactant2, size, rc.bindrad2,
								devGrid.indices, devGrid.offsets, devNearest, g_cpuParams.accuracy > 5);

	SecondReactionsKernelDev2<<<numBlocks,numThreads>>>(mols, col, rc.products[0], prod2,  size, c, c2, devNearest);

#endif
}

__host__	uint	DoSecondReactions(molecule_t*	mols, float3*	colors, uint size)
{
	cudaMemcpy(g_freeIdx, &size, sizeof(uint), cudaMemcpyHostToDevice);

	smolparams_t::secondreact_map_t::const_iterator it = g_cpuParams.secondreact.begin();
	smolparams_t::secondreact_map_t::const_iterator end = g_cpuParams.secondreact.end();

	uint	newSize = size;

	bool	doCompaction = false;

	for(;it != end; ++it)
	{
		if(it->second.deallocMolecules())
			doCompaction = true;

		RunSecondReactionsKernel(mols, colors, size, it->second);
/*
		if(it->second.deallocMolecules())
		{
			newSize = RemoveDeadMolecules(mols, colors, newSize);
			cudaMemcpy(g_freeIdx, &newSize, sizeof(uint), cudaMemcpyHostToDevice);
		}
*/
//		cudaMemcpy(&newSize, g_freeIdx, sizeof(uint), cudaMemcpyDeviceToHost);
	}

	if(doCompaction)
		newSize = RemoveDeadMolecules(mols, colors, newSize);

	return newSize;
}

__host__	uint	DoSecondReactionsNewHash(molecule_t*	mols, float3*	colors, uint size)
{
	cudaMemcpy(g_freeIdx, &size, sizeof(uint), cudaMemcpyHostToDevice);

	smolparams_t::secondreact_map_t::const_iterator it = g_cpuParams.secondreact.begin();
	smolparams_t::secondreact_map_t::const_iterator end = g_cpuParams.secondreact.end();

	uint	newSize = size;

	bool	doCompaction = false;

	for(;it != end; ++it)
	{
		if(it->second.deallocMolecules())
			doCompaction = true;

		RunSecondReactionsKernelNewHash(mols, colors, size, it->second);
	}

	if(doCompaction)
		newSize = RemoveDeadMolecules(mols, colors, newSize);

	return newSize;
}

__global__	void	CalculateCellIDsDev(molecule_t*	mols, uint* d_hashes, uint* d_indices, uint size)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= size)
		return;

	molecule_t	mol = mols[idx];

	d_hashes[idx] = GetCellID(mol.pos);

	d_indices[idx] = idx;
}

__host__	void	CalculateCellIDs(molecule_t*	mols, uint* d_hashes, uint* d_indices, uint size)
{
	uint	numThreads = 128;
	uint	numBlocks = GetNumberOfBlocks(numThreads, size);

	CalculateCellIDsDev<<<numBlocks,numThreads>>>(mols, d_hashes, d_indices, size);
}

__global__	void	FindCellBoundariesDev(uint2*   cellStartEnd, uint*  hashes, uint*  indices, uint size)
{
	extern __shared__ uint sharedHash[];

	uint idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= size)
		return;

	uint hash = hashes[idx];

	sharedHash[threadIdx.x+1] = hash;

	if (idx > 0 && threadIdx.x == 0)
	{
		sharedHash[0] = hashes[idx-1];
	}

	__syncthreads();
                                 //previous one
	if(idx == 0 || hash != sharedHash[threadIdx.x])
	{
		cellStartEnd[hash].x = idx;
		if (idx > 0)
			cellStartEnd[sharedHash[threadIdx.x]].y = idx;
	}

	if (idx == size - 1)
	{
		cellStartEnd[hash].y = idx + 1;
	}
}

__host__	void	FindCellBoundaries(uint2*   cellStartEnd,
										 uint *  gridParticleHash, // input: sorted grid hashes
										 uint *  gridParticleIndex,// input: sorted particle indices
										 uint numParticles)
{
	const uint	numThreads = 256;
    const uint	smemSize   = sizeof(uint)*(numThreads+1);
    const uint	numBlocks  = GetNumberOfBlocks(numThreads, numParticles);

    FindCellBoundariesDev<<<numBlocks, numThreads, smemSize>>>(cellStartEnd, gridParticleHash, gridParticleIndex, numParticles);
}

uint	CalculateSortingBits(uint gridCells)
{
	int	i = -1;
	while(gridCells)
	{
		gridCells>>=1;
		i++;
	}

	return i;
}

__host__	void	SortParticlesIndices(uint*	hashes, uint*	indices, uint size)
{
	if(radixSort == 0)
		radixSort = new nvRadixSort::RadixSort(g_cpuParams.max_mol);

	radixSort->sort(hashes, indices, size, g_sortBits);
}

void	HashMolecules(molecule_t*	mols, uint size)
{
	CalculateCellIDs(mols,devGrid.hashes,devGrid.indices, size);

	SortParticlesIndices(devGrid.hashes,devGrid.indices, size);

	FindCellBoundaries(devGrid.cellStartEnd,devGrid.hashes, devGrid.indices, size);
}

struct	MolHashFunc
{
	__device__	uint	operator()(const smolgpu::molecule_t& mol) const
	{
		return GetCellID(mol.pos);
	}
};

void	HashMoleculesAtomics(molecule_t*	mols, uint size)
{
	//template	<class	T, class P>
	//__host__	void	BuildGrid(T* mols, uint* offsets, int*	indices, int size, int numCells, P HashFunc)

	uint	numCells = devGrid.gridDim;

	numCells *= numCells*numCells;

	MolHashFunc	hashFunc;

	cudahelp::BuildGrid(mols, devGrid.offsets, devGrid.indices, size, numCells, hashFunc);
}

//default size is 4096
__host__	void	InitGaussTable()
{
	g_gaussTableSize = 4096;

	float*	cpuGaussTable = new float[g_gaussTableSize];

	GenerateGaussTable(cpuGaussTable, g_gaussTableSize);

	cudaMalloc((void**)&devGaussTable, g_gaussTableSize*sizeof(float));

	cudaMemcpy(devGaussTable, cpuGaussTable, g_gaussTableSize*sizeof(float), cudaMemcpyHostToDevice);

	delete [] cpuGaussTable;
}

__host__	void	InitDiffStates()
{
	int	statesCount = g_cpuParams.species.size()*STATES_COUNT;

	float*	cpuDiffTable = new float[statesCount];

	smolparams_t::species_map_t::const_iterator it = g_cpuParams.species.begin();
	smolparams_t::species_map_t::const_iterator end = g_cpuParams.species.end();

	//TODO: parallel kernel execution
	for(; it != end; ++it)
	{
		int idx = it->second.id*5;
		cpuDiffTable[idx + 0] = it->second.difc;
		cpuDiffTable[idx + 1] = it->second.difc;
		cpuDiffTable[idx + 2] = it->second.difc;
		cpuDiffTable[idx + 3] = it->second.difc;
		cpuDiffTable[idx + 4] = it->second.difc;
	}

	cudaMalloc((void**)&devDiffC, statesCount*sizeof(float));
	cudaMalloc((void**)&devDiffStep, statesCount*sizeof(float));

	cudaMemcpy(devDiffC, cpuDiffTable, statesCount*sizeof(float), cudaMemcpyHostToDevice);

	delete [] cpuDiffTable;
}

void		CreateGPUGrid();

__host__	void	InitSimulationParameters(const smolparams_t&	params)
{
	g_cpuParams = params;

	smol_gpu_params_t	p;
	p.timestep = params.timeStep;

	p.bmin = make_float3(params.boundaries.min.x,params.boundaries.min.y,params.boundaries.min.z);
	p.bmax = make_float3(params.boundaries.max.x,params.boundaries.max.y,params.boundaries.max.z);

	int d = params.getGridDim();

	p.dim = make_uint3(d,d,d);

	p.csize = params.csize;

	if(cudaSuccess != cudaMemcpyToSymbol(g_parameters, &p, sizeof(smol_gpu_params_t)))
	{
		printf("\nCan't init Smoldyn's simulation settings\n");
	}

	cudaMalloc((void**)&g_freeIdx, sizeof(uint));

	cudaMalloc((void**)&devTmpMolecules, params.max_mol*sizeof(molecule_t));
	cudaMalloc((void**)&devMomentums, params.max_mol*sizeof(momentum_t));

	cudaMalloc((void**)&devNearest, params.max_mol*sizeof(int));

	thrust::device_ptr<molecule_t> dev_data(devTmpMolecules);
	thrust::device_ptr<molecule_t> dev_data_end(devTmpMolecules + g_cpuParams.max_mol);

	molecule_t	deadMol = {make_float3(0,0,0), -1};

	thrust::fill(dev_data, dev_data_end, deadMol);

	InitGaussTable();
	InitDiffStates();
	CreateGPUGrid();

	cudahelp::InitMutexes(g_cpuParams.max_mol);
}
/*
 * struct	gpu_grid_t
{
	uint2*	cellStartEnd;
	uint* 	indices;
};

gpu_grid_t	devGrid;
 */
void		CreateGPUGrid()
{
	int	numCells = g_cpuParams.getGridDim();

	devGrid.gridDim = numCells;

	numCells *= numCells*numCells;

	g_sortBits = CalculateSortingBits(numCells);

	cudaMalloc((void**)&devGrid.cellStartEnd, numCells*sizeof(uint2));
	cudaMemset(devGrid.cellStartEnd,0,numCells*sizeof(uint2));

	cudaMalloc((void**)&devGrid.offsets, (numCells+1)*sizeof(uint));

	cudaMalloc((void**)&devGrid.indices, g_cpuParams.max_mol*sizeof(uint));
	cudaMalloc((void**)&devGrid.hashes, g_cpuParams.max_mol*sizeof(uint));

	cudahelp::InitGrid(g_cpuParams.max_mol, numCells);
}

/*
 * struct	diff_stat_t
{
	int	id;
	int	count;
	float3	avgPos;
	float3	m1; //xy, xz, yz//shear
	float3	m2; //x2, y2, z2//normal
};
 */

struct mol_plus_pred
{
	__host__ __device__
	mol_plus_pred() {}

	__host__ __device__
	smolgpu::molecule_t operator()(const smolgpu::molecule_t& m1, const smolgpu::molecule_t& m2)
	{
		molecule_t	m;
		m.type = m1.type;

		m.pos = make_float3(m1.pos.x+m2.pos.x, m1.pos.y+m2.pos.y, m1.pos.z+m2.pos.z);

		return m;
	}
};



struct momentum_plus_pred
{
	__host__ __device__
	momentum_plus_pred() {}

	__host__ __device__
	momentum_t operator()(const momentum_t& m1, const momentum_t& m2)
	{
		momentum_t	m;

		m.m1 = make_float3(m1.m1.x+m2.m1.x, m1.m1.y+m2.m1.y, m1.m1.z+m2.m1.z);
		m.m2 = make_float3(m1.m2.x+m2.m2.x, m1.m2.y+m2.m2.y, m1.m2.z+m2.m2.z);

		return m;
	}
};

struct pos2moment1
{
	__host__ __device__
	pos2moment1() {}

	__host__ __device__
	momentum_t	operator()(const smolgpu::molecule_t& m1)
	{
		momentum_t	m;
		m.m1 = m1.pos;

		return m;
	}
};

struct calc_momenta
{
	float3	avgPos;

	__host__ __device__
	calc_momenta(float3	av): avgPos(av) {}

	//m1 - pos
	__host__ __device__
	momentum_t	operator()(const momentum_t& m1)
	{
		momentum_t	m;

		m.m2.x = (m1.m1.x-avgPos.x)*(m1.m1.x-avgPos.x);
		m.m2.y = (m1.m1.y-avgPos.y)*(m1.m1.y-avgPos.y);
		m.m2.z = (m1.m1.z-avgPos.z)*(m1.m1.z-avgPos.z);

		m.m1.x = (m1.m1.x-avgPos.x)*(m1.m1.y-avgPos.y);
		m.m1.y = (m1.m1.x-avgPos.x)*(m1.m1.z-avgPos.z);
		m.m1.z = (m1.m1.y-avgPos.y)*(m1.m1.z-avgPos.z);

		return m;
	}
};

__host__	void	CalculateMolcountStatistics(std::vector<molcount_stat_t>& stat, molecule_t* mols, int actSize)
{
	smolparams_t::species_map_t::const_iterator it = g_cpuParams.species.begin();
	smolparams_t::species_map_t::const_iterator end = g_cpuParams.species.end();

	stat.clear();

	stat.resize(g_cpuParams.species.size());

	for( ;it != end; ++it)
	{
		molcount_stat_t s;

		s.id = it->second.id;

		thrust::device_ptr<molecule_t> dev_data(mols);
		thrust::device_ptr<molecule_t> dev_data_end(mols + actSize);

		s.count = thrust::count_if(dev_data, dev_data_end, mol_type_pred(s.id));

		stat[s.id] = s;
	}
}

struct	hist_condition_x
{
	hist_condition_x(float3 l, float	bsize):left(l), bin_size(bsize) {}
	__device__	int	operator()(float3	pos)
	{
		return (pos.x-left.x) / bin_size;
	}

	float	bin_size;
	float3	left;
};

struct	hist_condition_y
{
	hist_condition_y(float3 l, float	bsize):left(l), bin_size(bsize) {}
	__device__	int	operator()(float3	pos)
	{
		return (pos.y-left.y) / bin_size;
	}

	float	bin_size;
	float3	left;
};

struct	hist_condition_z
{
	hist_condition_z(float3 l, float	bsize):left(l), bin_size(bsize) {}
	__device__	int	operator()(float3	pos)
	{
		return (pos.z-left.z) / bin_size;
	}

	float	bin_size;
	float3	left;
};

__device__	bool	MolInBox(float3 pos, float3 bmin, float3 bmax)
{
	return  pos.x > bmin.x && pos.x < bmax.x &&
		pos.y > bmin.y && pos.y < bmax.y &&
		pos.z > bmin.z && pos.z < bmax.z;
}

template	<class	HistAxis>
__global__	void	CalculateCountSpaceDev(uint*	hist, molecule_t* mols, int actSize, int specId, float3 bmin, float3 bmax, HistAxis axis)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= actSize)
		return;

	molecule_t	mol = mols[idx];

	if(specId == -1 || (mol.type == specId && specId != -1))
	{
		if(MolInBox(mol.pos, bmin, bmax))
		{
			int b = axis(mol.pos);
			atomicAdd(&hist[b],1);
		}
	}
}

__host__	void	CalculateMolCountSpace(std::vector<int>& v, molecule_t* mols, int actSize, int axis, int bins, int specId, float3 bmin, float3 bmax)
{
	if(g_histSize < bins)
	{
		if(devHistogram)
			cudaFree(devHistogram);

		cudaMalloc((void**)&devHistogram, bins*sizeof(uint));

		g_histSize = bins;
	}

	cudaMemset(devHistogram, 0, bins*sizeof(uint));

	int	numThreads = 128;
	int	numBlocks  = GetNumberOfBlocks(numThreads, actSize);

	switch(axis)
	{
	case 0:
		CalculateCountSpaceDev<<<numBlocks,numThreads>>>(devHistogram, mols, actSize, specId, bmin, bmax, hist_condition_x(bmin,(bmax.x-bmin.x)/bins));
		break;
	case 1:
		CalculateCountSpaceDev<<<numBlocks,numThreads>>>(devHistogram, mols, actSize, specId, bmin, bmax, hist_condition_y(bmin,(bmax.y-bmin.y)/bins));
		break;
	case 2:
		CalculateCountSpaceDev<<<numBlocks,numThreads>>>(devHistogram, mols, actSize, specId, bmin, bmax, hist_condition_z(bmin,(bmax.z-bmin.z)/bins));
		break;
	}

	cudaMemcpy(&v[0], devHistogram, bins*sizeof(uint), cudaMemcpyDeviceToHost);
}

__global__	void	ProcessEquimolDev(molecule_t* mols, float3* colors, int size, float3 c1, float3 c2, int spec1, int spec2, float prob)
{
	const uint idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= size)
		return;

	molecule_t	mol = mols[idx];

	if(mol.type == spec1 || mol.type == spec2)
	{
		cudahelp::rand::Xor128Generator gen(idx);

		if(gen.GetFloat() < 1-prob)
		{
			mol.type = spec1;
			colors[idx] = c1;
		}
		else
		{
			mol.type = spec2;
			colors[idx] = c2;
		}

		mols[idx] = mol;
	}

}

__host__	void	ProcessEquimol(molecule_t* mols, float3* colors, int size, int spec1, int spec2, float prob)
{
	int	numThreads = 128;
	int	numBlocks  = GetNumberOfBlocks(numThreads, size);

	float3	c1 = g_cpuParams.getSpiece(spec1).color;
	float3	c2 = g_cpuParams.getSpiece(spec2).color;

	ProcessEquimolDev<<<numBlocks,numThreads>>>(mols, colors, size, c1, c2, spec1, spec2, prob);
}

__host__	void	CalculateDiffStatistics(std::vector<diff_stat_t>& stat, molecule_t*	mols, int actSize)
{
	smolparams_t::species_map_t::const_iterator it = g_cpuParams.species.begin();
	smolparams_t::species_map_t::const_iterator end = g_cpuParams.species.end();

	for(int	i = 0; it != end; ++it, i++)
	{
		diff_stat_t d;

		thrust::device_ptr<molecule_t> dev_data(mols);
		thrust::device_ptr<molecule_t> dev_data_end(mols + actSize);
		thrust::device_ptr<molecule_t> dev_tmp_data(devTmpMolecules);
		thrust::device_ptr<momentum_t> dev_momentum_data(devMomentums);

		d.id = it->second.id;

		d.count = 0;

		d.count = thrust::copy_if(dev_data, dev_data_end, dev_tmp_data, mol_type_pred(d.id)) - dev_tmp_data;

		thrust::device_ptr<molecule_t> dev_tmp_data_end(devTmpMolecules + d.count);

		thrust::transform(dev_tmp_data, dev_tmp_data_end, dev_momentum_data, pos2moment1());

		inclusive_scan(dev_tmp_data, dev_tmp_data_end, dev_tmp_data, mol_plus_pred());

		molecule_t	m;

		cudaMemcpy(&m, devTmpMolecules+d.count-1, sizeof(molecule_t), cudaMemcpyDeviceToHost);

		m.pos.x /= d.count;
		m.pos.y /= d.count;
		m.pos.z /= d.count;

		d.avgPos = m.pos;

		thrust::device_ptr<momentum_t> dev_momentum_data_end(devMomentums + d.count);

		thrust::transform(dev_momentum_data, dev_momentum_data_end, dev_momentum_data, calc_momenta(d.avgPos));

		inclusive_scan(dev_momentum_data, dev_momentum_data_end, dev_momentum_data, momentum_plus_pred());

		momentum_t	mm;

		cudaMemcpy(&mm, devMomentums+d.count-1, sizeof(momentum_t), cudaMemcpyDeviceToHost);

		//momenta: transform -> reduce

		mm.m1.x /= d.count;
		mm.m1.y /= d.count;
		mm.m1.z /= d.count;

		mm.m2.x /= d.count;
		mm.m2.y /= d.count;
		mm.m2.z /= d.count;

		d.m1 = mm.m1;
		d.m2 = mm.m2;

		stat[i] = d;
	}

}

__host__	void	DeinitSpecies()
{
	if(devMomentums)
		cudaFree(devMomentums);

	if(devNearest)
		cudaFree(devNearest);

	if(devTmpMolecules)
		cudaFree(devTmpMolecules);

	if(devDiffC)
		cudaFree(devDiffC);

	if(devDiffStep)
		cudaFree(devDiffStep);

	if(devGaussTable)
		cudaFree(devGaussTable);

	if(devGrid.cellStartEnd)
		cudaFree(devGrid.cellStartEnd);

	if(devGrid.indices)
		cudaFree(devGrid.indices);

	if(devGrid.offsets)
		cudaFree(devGrid.offsets);

	if(devGrid.hashes)
		cudaFree(devGrid.hashes);

	if(radixSort)
		delete radixSort;

	cudahelp::DeinitMutexes();

	cudahelp::DeinitGrid();
}

__host__	void	ProcessBoundaryConditions(molecule_t*	mols, uint molsAlive)
{
	float3	bmin = make_float3(g_cpuParams.boundaries.min.x, g_cpuParams.boundaries.min.y, g_cpuParams.boundaries.min.z);
	float3	bmax = make_float3(g_cpuParams.boundaries.max.x, g_cpuParams.boundaries.max.y, g_cpuParams.boundaries.max.z);

	ProcessBoundaryConditionsLookup(mols, molsAlive, bmin, bmax, g_cpuParams.boundCond);
}
/*
const int N = 6;
  int A[N] = {1, 4, 2, 8, 5, 7};
  int *new_end = thrust::remove(A, A + N, is_even());
 */
__host__	uint	RemoveDeadMolecules(molecule_t*	mols, float3* color, uint size)
{
	thrust::device_ptr<molecule_t> dev_data(mols);
	thrust::device_ptr<molecule_t> dev_data_end(mols + size);

	thrust::device_ptr<float3> dev_cols(color);
	thrust::device_ptr<float3> dev_cols_end(color + size);

	thrust::remove_if(dev_cols, dev_cols_end, dev_data, mol_type_pred(-1));

	uint newSize = thrust::remove_if(dev_data, dev_data_end, mol_type_pred(-1)) - dev_data;

	return newSize;
}

__host__	uint	CalculateDiffusion(molecule_t*	mols, float3* colors, uint molsAlive)
{
	int	numThreads = 128;
	int	numBlocks  = GetNumberOfBlocks(numThreads, molsAlive);

	CalculateDiffusionDev<<<numBlocks, numThreads>>>(mols,devDiffStep, devGaussTable, g_gaussTableSize, molsAlive);

	ProcessBoundaryConditions(mols, molsAlive);

	if(g_cpuParams.boundCond[0] == 'a' || g_cpuParams.boundCond[1] == 'a' || g_cpuParams.boundCond[2] == 'a')
		return RemoveDeadMolecules(mols, colors,molsAlive);

	return molsAlive;
}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <float.h>

#define PI 3.14159265358979323846
#define SQRT2 1.41421356237
#define SQRTPI 1.7724538509
#define SQRT2PI 2.50662827462
/*
inline static float randCCF(void) {
	return (float)rand()*(1.0/RAND_MAX); }
*/

#define RAND_BITS 30
	#define RAND_MAX_30 1073741823UL
	#define RAND2_MAX 1073741823UL
	#if RAND_MAX==32767
		#define rand30() ((unsigned long int)rand()<<15|(unsigned long int)rand())
	#elif RAND_MAX<RAND_MAX_30
		#define rand30() (((unsigned long int)rand()&32767UL)<<15|((unsigned long int)rand()&32767UL))
	#else
		#define rand30() ((unsigned long int)rand()&RAND_MAX_30)
	#endif

	inline static double randCCD(void) {
		return (double)rand30()*(1.0/RAND_MAX_30); }

float gammaln(float x)	{	/* Returns natural log of gamma function, partly from Numerical Recipies and part from Math CRC */
	double sum,t;
	int j;
	static double c[6]={76.18009173,-86.50532033,24.01409822,-1.231739516,0.120858003e-2,-0.536382e-5};

	if(x==floor(x)&&x<=0) sum=DBL_MAX;					// 0 or negative integer
	else if(x==floor(x))	{											// positive integer
		sum=0;
		for(t=2;t<x-0.1;t+=1.0)	sum+=log(t);	}
	else if(x==0.5)	sum=0.572364942;						// 1/2
	else if(2*x==floor(2*x)&&x>0)	{							// other positive half integer
		sum=0.572364942;
		for(t=0.5;t<x-0.1;t+=1)	sum+=log(t);	}
	else if(2*x==floor(2*x))	{									// negative half integer
		sum=0.572364942;
		for(t=0.5;t<-x+0.1;t+=1)	sum-=log(t);	}
	else if(x<0)																// other negative
		sum=gammaln(x+1)-log(-x);
	else	{																			// other positive
		x-=1.0;
		t=x+5.5;
		t-=(x+0.5)*log(t);
		sum=1.0;
		for(j=0;j<=5;j++)	{
			x+=1.0;
			sum+=c[j]/x;	}
		sum=-t+log(2.50662827465*sum);	}
	return(sum);	}

int poisrandD(double xm) {
	static double sq,alxm,g,oldm=-1.0;
	float em,t,y;

	if(xm<=0) return 0;
	else if(xm<12.0) {
		if(xm!=oldm) {
			oldm=xm;
			g=exp(-xm); }
		em=0;
		for(t=randCCD();t>g;t*=randCCD()) {
			em+=1.0; }}
	else {
		if(xm!=oldm) {
			oldm=xm;
			sq=sqrt(2.0*xm);
			alxm=log(xm);
			g=xm*alxm-gammaln(xm+1.0); }
		do {
			do {
				y=tan(PI*randCCD());
				em=sq*y+xm; } while(em<0);
			em=floor(em);
			t=0.9*(1.0+y*y)*exp(em*alxm-gammaln(em+1.0)-g); } while(randCCD()>t); }
	return (int) em; }

/*
int poisrandF(float xm) {
	static float sq,alxm,g,oldm=-1.0;
	float em,t,y;

	if(xm<=0) return 0;
	else if(xm<12.0) {
		if(xm!=oldm) {
			oldm=xm;
			g=exp(-xm); }
		em=0;
		for(t=randCCF();t>g;t*=randCCF()) {
			em+=1.0; }}
	else {
		if(xm!=oldm) {
			oldm=xm;
			sq=sqrt(2.0*xm);
			alxm=log(xm);
			g=xm*alxm-gammaln(xm+1.0); }
		do {
			do {
				y=tan(PI*randCCF());
				em=sq*y+xm; } while(em<0);
			em=floor(em);
			t=0.9*(1.0+y*y)*exp(em*alxm-gammaln(em+1.0)-g); } while(randCCF()>t); }
	return (int) em; }
*/
float gammp(float a,float x)	{ // Returns incomplete gamma function, partly from Numerical Recipes
	double sum,del,ap,eps;
	double gold=0,g=1,fac=1,b1=1,b0=0,anf,ana,an=0,a1,a0=1;

	eps=3e-7;
	if(x<0||a<=0) return -1;			// out of bounds
	else if(x==0) return 0;
	else if(x<a+1)	{
		ap=a;
		del=sum=1.0/a;
		while(fabs(del)>fabs(sum)*eps&&ap-a<100)	{
			ap+=1;
			del*=x/ap;
			sum+=del;	}
		return sum*exp(-x+a*log(x)-gammaln(a));	}
	else {
		a1=x;
		for(an=1;an<100;an++)	{
			ana=an-a;
			a0=(a1+a0*ana)*fac;
			b0=(b1+b0*ana)*fac;
			anf=an*fac;
			a1=x*a0+anf*a1;
			b1=x*b0+anf*b1;
			if(a1)	{
				fac=1.0/a1;
				g=b1*fac;
				if(fabs((g-gold)/g)<eps)
					return 1.0-exp(-x+a*log(x)-gammaln(a))*g;
				gold=g;	}}}
	return -1;	}							// shouldn't ever get here

float erfn(float x)	{				// Numerical Recipies
	return (x<0?-gammp(0.5,x*x):gammp(0.5,x*x));	}


float erfcc(float x) {
	double t,z,ans;

	z=fabs(x);
	t=1.0/(1.0+0.5*z);
	ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
	return x>=0.0?ans:2.0-ans; }

float erfcintegral(float x) {
	return (1.0-exp(-x*x))/SQRTPI+x*erfcc(x); }


float inversefn(float (*fn)(float),float y,float x1,float x2,int n) {
	float dx,y2;

	if((*fn)(x1)<(*fn)(x2))	dx=x2-x1;
	else {
		dx=x1-x2;
		x1=x2; }
	for(;n>0;n--) {
		dx*=0.5;
		x2=x1+dx;
		y2=(*fn)(x2);
		if(y2<y) x1=x2; }
	return x1+0.5*dx; }

void randtableF(float *a,int n,int eq)
{
	int i;
	float dy;

	if(eq==1) {
		dy=2.0/n;
	  for(i=0;i<n/2;i++)
	    a[i]=SQRT2*inversefn(erfn,(i+0.5)*dy-1.0,-20,20,30);
	  for(i=n/2;i<n;i++)
	    a[i]=-a[n-i-1]; }
	else if(eq==2) {
		dy=1.0/SQRTPI/n;
		for(i=0;i<n;i++)
			a[i]=SQRT2*inversefn(erfcintegral,(i+0.5)*dy,0,20,30);
	}
}

void	GenerateGaussTable(float*	data, int size)
{
	randtableF(data, size, 1);
}

}

#include "geom_kernels.cu"
