/*
 * aabb.cpp
 *
 *  Created on: Mar 15, 2010
 *      Author: denis
 */

#include "aabb.h"

#include "math.h"

#include "tritest.h"

bbox_t	bbox_t::from_aabb(const	AABB_t& aabb)
{
	bbox_t	bbox;

	bbox.min = aabb.center - aabb.half_size;
	bbox.max = aabb.center + aabb.half_size;

	return bbox;
}

bool	AABB_t::tri_overlaps(const vec3_t& a, const vec3_t& b, const vec3_t& c) const
{
	float	bc[] = {center.x,center.y, center.z};
	float	hsz[] = {half_size.x, half_size.y, half_size.z};

	float	tris[][3] = {
			{a.x,a.y,a.z},
			{b.x,b.y,b.z},
			{c.x,c.y,c.z}
	};

	return triBoxOverlap(bc,hsz,tris) == 1;
}
