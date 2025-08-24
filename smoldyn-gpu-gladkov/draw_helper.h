/*
 * draw_helper.h
 *
 *  Created on: May 25, 2010
 *      Author: denis
 */

#ifndef DRAW_HELPER_H_
#define DRAW_HELPER_H_

#include "config.h"
#include "reggrid.h"

#include "model.h"

struct	mouse_state_t
{
	bool	leftBtnPressed;
	bool	rightBtnPressed;
	int		pos_x;
	int		pos_y;
};

struct	simulation_state_t
{
	bool	paused;
	bool	drawGrid;
	bool	drawVectorField;

	bool	demoMode;

	bool	processCollisions;

	bool	sampleConcentration;

	float	simTime;
	float	realTime;

	uint	slice;

	float	fps;

	uint	stepCount;
	uint	samplesCount;

	simulation_state_t(): paused(true), drawGrid(false), drawVectorField(false), demoMode(false),
							processCollisions(true), simTime(0.0f), realTime(0.0f), slice(0), fps(0), stepCount(0) {}

	void	togglePause() {paused = !paused;}
	void	toggleDrawGrid() {drawGrid = !drawGrid;}
	void	toggleDrawVectorField() {drawVectorField = !drawVectorField;}
	void	toggleDemoMode() {demoMode = !demoMode;}
	void	toggleCollisions() {processCollisions = !processCollisions;}
	void	toggleConcentration() {sampleConcentration = !sampleConcentration;}
	void	advanceSimTime(float dt) {simTime += dt;}
	void	advanceRealTime(float dt) {realTime += dt;}
	void	advanceSimStep() {stepCount++;}
	void	advanceSamplesCount() {samplesCount++;}

	void	incSlice(uint dimz);
	void	decSlice();
};


void	setup2DView(int	w, int h);
void	DrawText(float x, float y, const char* str);
void	DrawText2D(uint x, uint y, const char* str);
void	DrawBBox(const bbox_t& bbox);
void	DrawBBoxColored(const bbox_t& bbox);
void	DrawAABB(const	AABB_t&	aabb);
void	DrawRegularGrid(reg_grid_t& grid);
void	DrawSimulationGrid(const bbox_t& gbbox, const vec3i_t& dim, int slice = -1);
void	DrawMatrix(uint*	data, vec3i_t dim, uint slice);
uint	CreateTexture(unsigned int size_x, unsigned int size_y);
uint	CreateBufferObject(unsigned int size, bool colorBuffer = false);
uint	CreateCUDABufferObject(unsigned int size, bool colorBuffer = false);

void	RenderModel(Model&, bool norm = true);

void	ReshapePerspective(uint w, uint h);
void	ReshapePerspective(uint w, uint h, vec3_t mn, vec3_t mx);

#endif /* DRAW_HELPER_H_ */
