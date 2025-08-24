/*
 * particles.cpp
 *
 *  Created on: Jun 1, 2010
 *      Author: denis
 */

#include "molecules.h"

#include "draw_helper.h"

#include <stdexcept>

#include <GL/glew.h>

#define FREEGLUT_LIB_PRAGMAS 0
#define FREEGLUT_STATIC 1
#include <GL/glut.h>

#include <cuda_gl_interop.h>

namespace smolgpu
{
/*
struct	base_mols_t
{
	uint	maxMolecules;
	uint	aliveMolecules;
	bool*	isAlive;

	base_mols_t(): maxMolecules(0), aliveMolecules(0), isAlive(0) {}
	~base_mols_t();

	void	init(uint maxSamp);
};
*/

	base_mols_t::~base_mols_t()
	{
		if(isAlive)
			cudaFree(isAlive);
	}

	void	base_mols_t::init(uint maxSamp)
	{
		maxMolecules = maxSamp;

		cudaMalloc((void**)&isAlive, maxSamp*sizeof(uint));
		cudaMemset(isAlive, 0, maxSamp*sizeof(uint));
	}

	molecule_state_t::~molecule_state_t()
	{
		if(colors)
			cudaFree(colors);

		if(molecules)
			cudaFree(molecules);
	}

	void	molecule_state_t::init(uint samples)
	{
		base_mols_t::init(samples);

		cudaMalloc((void**)&molecules, samples*sizeof(molecule_t));
		cudaMalloc((void**)&colors, samples*sizeof(float3));
	}

	void	molecule_state_t::render()
	{
		glVertexPointer(3, GL_FLOAT, 16, molecules);
		glEnableClientState(GL_VERTEX_ARRAY);

		glColorPointer(3, GL_FLOAT, 0, colors);
		glEnableClientState(GL_COLOR_ARRAY);

		glDrawArrays(GL_POINTS, 0, aliveMolecules);

		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

//
//
//


	molecule_state_vbo_t::~molecule_state_vbo_t()
	{
		cudaGLUnregisterBufferObject(moleculesVBO);
		glDeleteBuffers(1, &moleculesVBO);

		cudaGLUnregisterBufferObject(colorsVBO);
		glDeleteBuffers(1, &colorsVBO);

		if(prev_mols)
			cudaFree(prev_mols);
	}

	molecule_t*	molecule_state_vbo_t::lock_molecules()
	{
		molecule_t*	mols = 0;

		cudaGLMapBufferObject((void**)&mols, moleculesVBO);

		return mols;
	}

	void	molecule_state_vbo_t::unlock_molecules()
	{
		cudaGLUnmapBufferObject(moleculesVBO);
	}

	float3*	molecule_state_vbo_t::lock_colors()
	{
		float3*	colors = 0;

		cudaGLMapBufferObject((void**)&colors, colorsVBO);

		return colors;
	}

	void	molecule_state_vbo_t::unlock_colors()
	{
		cudaGLUnmapBufferObject(colorsVBO);
	}

	void	molecule_state_vbo_t::copy_prev_mols(molecule_t* curr)
	{
		if(curr == 0)
		{
			molecule_t*	cc = lock_molecules();
			cudaMemcpy(prev_mols,cc,maxMolecules*sizeof(molecule_t), cudaMemcpyDeviceToDevice);
			unlock_molecules();
		}else
			cudaMemcpy(prev_mols,curr,maxMolecules*sizeof(molecule_t), cudaMemcpyDeviceToDevice);
	}

	void	molecule_state_vbo_t::init(uint samples)
	{
		base_mols_t::init(samples);

		uint size = samples*sizeof(molecule_t);

		moleculesVBO	= CreateCUDABufferObject(size);

		size = samples*sizeof(float3);

		colorsVBO = CreateCUDABufferObject(size);

		cudaMalloc(&prev_mols, samples*sizeof(molecule_t));
	}

	void	molecule_state_vbo_t::render()
	{
		glBindBuffer(GL_ARRAY_BUFFER, moleculesVBO);

		glVertexPointer(3, GL_FLOAT, 16, 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
		glColorPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);

		glDrawArrays(GL_POINTS, 0, aliveMolecules);

		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
}
