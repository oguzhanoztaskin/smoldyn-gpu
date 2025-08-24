/*
 * draw_helper.cpp
 *
 *  Created on: May 25, 2010
 *      Author: denis
 */

#include "draw_helper.h"
#include "dsmc.h"
#include <stdio.h>

#include <GL/glew.h>

#define FREEGLUT_STATIC 1
#define FREEGLUT_LIB_PRAGMAS 0
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void	simulation_state_t::incSlice(uint dimz) {if(slice < dimz-1) slice++;}
void	simulation_state_t::decSlice() {if(slice > 0) slice--;}

void	setup2DView(int	w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0,w,h,0,-100,100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void	DrawText(float x, float y, const char* str)
{
	glRasterPos3f(x,y,-1);
	while(*str)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,*str++);

}

void	DrawText2D(uint x, uint y, const char* str)
{
	glRasterPos2i(x,y);
	while(*str)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,*str++);

}

void	DrawBBox(const bbox_t& bbox)
{
	glBegin(GL_LINE_STRIP);

		glVertex3f(bbox.min.x,bbox.min.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.min.z);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.min.z);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.min.z);

	glEnd();

	glBegin(GL_LINE_STRIP);

		glVertex3f(bbox.min.x,bbox.min.y,bbox.max.z);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.max.z);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.max.z);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.max.z);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.max.z);

	glEnd();

	glBegin(GL_LINES);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.max.z);

		glVertex3f(bbox.min.x,bbox.max.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.max.z);

		glVertex3f(bbox.max.x,bbox.max.y,bbox.min.z);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.max.z);

		glVertex3f(bbox.max.x,bbox.min.y,bbox.min.z);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.max.z);
	glEnd();
}

void	DrawBBoxColored(const bbox_t& bbox)
{
#define	LOW_COLOR 0x33,0x00,0xFF
#define HIGH_COLOR 0x33,0xFF,0x00

	glEnable(GL_LINE_SMOOTH);

	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glColor3ub(LOW_COLOR);
	glBegin(GL_LINE_STRIP);

		glVertex3f(bbox.min.x,bbox.min.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.max.z);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.max.z);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.min.z);

	glEnd();

	glColor3ub(HIGH_COLOR);
	glBegin(GL_LINE_STRIP);

		glVertex3f(bbox.min.x,bbox.max.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.max.z);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.max.z);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.min.z);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.min.z);

	glEnd();

	glBegin(GL_LINES);

		glColor3ub(HIGH_COLOR);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.min.z);
		glColor3ub(LOW_COLOR);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.min.z);

		glColor3ub(HIGH_COLOR);
		glVertex3f(bbox.min.x,bbox.max.y,bbox.max.z);
		glColor3ub(LOW_COLOR);
		glVertex3f(bbox.min.x,bbox.min.y,bbox.max.z);

		glColor3ub(HIGH_COLOR);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.max.z);
		glColor3ub(LOW_COLOR);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.max.z);

		glColor3ub(HIGH_COLOR);
		glVertex3f(bbox.max.x,bbox.max.y,bbox.min.z);
		glColor3ub(LOW_COLOR);
		glVertex3f(bbox.max.x,bbox.min.y,bbox.min.z);

	glEnd();

	glDisable(GL_LINE_SMOOTH);
}

void	DrawAABB(const	AABB_t&	aabb)
{
	DrawBBox(bbox_t::from_aabb(aabb));
}

void	DrawRegularGrid(reg_grid_t& grid)
{
	vec3i_t	gridDim = grid.dim;

	vec3i_t	gridIdx;

	for(gridIdx.x = 0; gridIdx.x < gridDim.x; gridIdx.x++)
		for(gridIdx.y = 0; gridIdx.y < gridDim.y; gridIdx.y++)
			for(gridIdx.z = 0; gridIdx.z < gridDim.z; gridIdx.z++)
			{
				AABB_t	aabb = grid.getAABB(gridIdx);
				if(grid[gridIdx].second == true)
					glColor3f(0,1,0);
				else
					glColor3f(1,0,0);

				DrawAABB(aabb);
			}

}

void	DrawSimulationGrid(const bbox_t& gbbox, const vec3i_t& dim, int slice)
{
	vec3i_t	gridDim = dim;

	vec3i_t	gridIdx;

	vec3_t	cellSize((gbbox.max.x-gbbox.min.x)/dim.x,(gbbox.max.y-gbbox.min.y)/dim.y,(gbbox.max.z-gbbox.min.z)/dim.z);

	bbox_t	bbox;

	vec3_t mn, mx;

	if(slice != -1)
	{
		gridIdx.z = slice;
		gridDim.z = slice+1;
	}

	for(; gridIdx.z < gridDim.z; gridIdx.z++)
		for(gridIdx.x = 0; gridIdx.x < gridDim.x; gridIdx.x++)
			for(gridIdx.y = 0; gridIdx.y < gridDim.y; gridIdx.y++)
			{
				mn.reset(gbbox.min.x + cellSize.x*gridIdx.x,gbbox.min.y + cellSize.y*gridIdx.y,gbbox.min.z + cellSize.z*gridIdx.z);
				mx.reset(mn.x + cellSize.x,mn.y + cellSize.y, mn.z + cellSize.z);

				bbox.reset(mn,mx);

				DrawBBox(bbox);
			}
}

uint	CreateTexture(unsigned int size_x, unsigned int size_y)
{
	GLuint textureID = 0;
	// Enable Texturing
	glEnable(GL_TEXTURE_2D);


	 // Generate a texture identifier
	glGenTextures(1, &textureID);


	// Make this the current texture (remember that GL is state-based)
	glBindTexture( GL_TEXTURE_2D, textureID);


	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, size_x, size_y, 0,
		GL_RGB,GL_UNSIGNED_BYTE, NULL);


	// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

	glBindTexture( GL_TEXTURE_2D, 0);

	return textureID;
}

uint	CreateBufferObject(unsigned int size, bool colorBuffer)
{
	GLuint	buffId = 0;

	glGenBuffers(1, &buffId);

	GLenum	type = GL_ARRAY_BUFFER, drv = GL_DYNAMIC_DRAW;

	if(colorBuffer)
	{
		type = GL_PIXEL_UNPACK_BUFFER;
		drv = GL_DYNAMIC_COPY;
	}

	glBindBuffer(type, buffId);
	glBufferData(type, size, 0, drv);
	glBindBuffer(type, 0);

	return	buffId;
}

uint	CreateCUDABufferObject(unsigned int size, bool colorBuffer)
{
	GLuint	buffId = CreateBufferObject(size, colorBuffer);

	if(cudaSuccess != cudaGLRegisterBufferObject(buffId))
		printf("CreateCUDABufferObject failed!\n");

	return	buffId;
}

void	RenderModel(Model& model, bool norm)
{
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	GLfloat mat_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };

	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, mat_diffuse);

	glColor3f(1,1,1);

	glPushMatrix();

//	glScalef(0.9f,0.9f,1.0f);

	model.Render(norm);

	glPopMatrix();

	glDisable(GL_LIGHTING);
}
/*
 * void glFrustum(	GLdouble  	left,
 	GLdouble  	right,
 	GLdouble  	bottom,
 	GLdouble  	top,
 	GLdouble  	nearVal,
 	GLdouble  	farVal);
 *
 */
void	ReshapePerspective(uint w, uint h, vec3_t mn, vec3_t mx)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glViewport(0, 0, w, h);

	float	fplane = std::max(1000.0f, (mx.z - mn.z)*5);

	gluPerspective(45,(float)w / h,1, fplane);

	glMatrixMode(GL_MODELVIEW);
}

void	ReshapePerspective(uint w, uint h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glViewport(0, 0, w, h);

	gluPerspective(45,(float)w / h,1,100);
	glMatrixMode(GL_MODELVIEW);
}
