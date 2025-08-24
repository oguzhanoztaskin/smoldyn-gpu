/*
 * aabb.h
 *
 *  Created on: Mar 15, 2010
 *      Author: denis
 */

#ifndef AABB_H_
#define AABB_H_

#include "config.h"
#include "vector.h"

struct	AABB_t;

struct	bbox_t
{
	vec3_t	min;
	vec3_t	max;

	bbox_t() {}

	bbox_t(const vec3_t& a, const vec3_t& b)
	{
		reset(a,b);
	}

	static	bbox_t	from_aabb(const	AABB_t& aabb);

	float	get_xsize() { return max.x - min.x; }
	float	get_ysize() { return max.y - min.y; }
	float	get_zsize() { return max.z - min.z; }

	float	get_vol() { return get_xsize()*get_ysize()*get_zsize(); }

	void	reset(const vec3_t& a, const vec3_t& b)
	{
		min = a;
		max = b;
	}
};

struct	AABB_t
{
	vec3_t	center;
	vec3_t	half_size;

	AABB_t() {}

	AABB_t(const vec3_t& c, const vec3_t& size)
	{
		reset(c,size);
	}

	static	AABB_t	from_bbox(const	bbox_t& bbox)
	{
		AABB_t	a;

		a.half_size = (bbox.max - bbox.min)/2;
		a.center = bbox.min + a.half_size;

		return a;
	}

	void	reset(const vec3_t& c, const vec3_t& size)
	{
		center = c;
		half_size = size;
	}

	bool	tri_overlaps(const vec3_t& a, const vec3_t& b, const vec3_t& c) const;
};

#endif /* AABB_H_ */
