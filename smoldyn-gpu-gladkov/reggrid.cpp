/*
 * reggrid.cpp
 *
 *  Created on: Mar 16, 2010
 *      Author: denis
 */

#include "reggrid.h"

#include "stdio.h"

void	reg_grid_t::reset(const	bbox_t& size, const vec3i_t&	dm)
{
	printf("Grid size min: %f %f %f\n", size.min.x, size.min.y, size.min.z);
	printf("Grid size max: %f %f %f\n", size.max.x, size.max.y, size.max.z);

	gridSize = size;
	dim		 = dm;

	vec3_t gridWidth = gridSize.max - gridSize.min;

	cellSize.x = gridWidth.x / dim.x;
	cellSize.y = gridWidth.y / dim.y;
	cellSize.z = gridWidth.z / dim.z;

	nodes.resize(cellCount());
	normals.resize(cellCount());

	for(int i = 0; i < nodes.size(); i++)
		nodes[i].second = false;

}

bool	reg_grid_t::needClosure(uint x, uint y)
{
	vec3i_t	gridIdx(x,y,0);
	bool	b = false;
	uint	len = 0;
	for(;gridIdx.z < dim.z; gridIdx.z++)
	{
		if((*this)[gridIdx].first.size() > 0)
		{
			if(len > 0)
				return true;

			b = true;
			len = 0;
		}

		if(b && (*this)[gridIdx].first.size() == 0)
			len++;
	}

	return false;
}

void	reg_grid_t::encloseGrid()
{
	vec3i_t	gridIdx;

	for(gridIdx.x = 0; gridIdx.x < dim.x; gridIdx.x++)
	{
		for(gridIdx.y = 0; gridIdx.y < dim.y; gridIdx.y++)
		{
			if(!needClosure(gridIdx.x,gridIdx.y))
				continue;

			bool	bFill = false;
			uint	oneLen = 0;
			for(gridIdx.z = 0; gridIdx.z < dim.z; gridIdx.z++)
			{
				if((*this)[gridIdx].first.size() > 0)
					oneLen++;

				if((*this)[gridIdx].first.size() == 0)
					oneLen = 0;

				if((*this)[gridIdx].first.size() > 0 && oneLen == 1)
					bFill = !bFill;

				if(bFill)
					(*this)[gridIdx].second = true;
			}
		}
	}
}

AABB_t	reg_grid_t::getAABB(const vec3i_t&	idx)
{
	bbox_t	bbox;
	bbox.min = gridSize.min + vec3_t(idx.x*cellSize.x, idx.y*cellSize.y, idx.z*cellSize.z);
	bbox.max = bbox.min + cellSize;

	return AABB_t::from_bbox(bbox);
}

uint	reg_grid_t::getCellIdx(const vec3i_t& idx)
{
	return	dim.x*dim.y*idx.z + dim.x*idx.y + idx.x;
}
