#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <sstream>
#include <iomanip>

#include <fstream>

#include <iostream>

#include <stdexcept>

#include <GL/glew.h>

#define FREEGLUT_LIB_PRAGMAS 0
#define FREEGLUT_STATIC 1
#include <GL/glut.h>

#include "draw_helper.h"

#include "smol_solver.h"

#include "texture.h"

#include "smolparser.h"

#include "image_helper.h"

#include <cuda_gl_interop.h>

#include <vector_types.h>
#include <vector_functions.h>

#include <cuda_runtime.h>

#include "log_file.h"

#include "file_path.h"

const	char*	g_path = "files/";

char*	g_prefix = 0;

using namespace	smolgpu;

uint	g_width;
uint	g_height;

unsigned	char*	frameBuffer = 0;

int	g_interval = -1;

bool	g_webService = false;

bool	wiredFrame = false;

smoldyn::Texture	pointSprite;

//TODO: move to frame object
float3	position = make_float3(0,0,-37);
float3	rotation = make_float3(45.0f,45.0f,0);

mouse_state_t		g_mouse_state;
simulation_state_t	g_simState;
smolparams_t		g_settings;

float3	g_streamVel = make_float3(0.0,0,0);

//TODO: move to benchmarking object
cudaEvent_t start, stop;
float	gpuTime = 0;

smolgpu::SmoldynSolver*	solver;

std::string	screenShotName = "";
bool	takeScreenshot = false;

void	reshape(int w, int h)
{
	g_width = w;
	g_height = h;

	vec3_t	mn(g_settings.boundaries.min.x,g_settings.boundaries.min.y,g_settings.boundaries.min.z);
	vec3_t	mx(g_settings.boundaries.max.x,g_settings.boundaries.max.y,g_settings.boundaries.max.z);

	ReshapePerspective(w, h, mn, mx);

//	ReshapePerspective(w, h);
}

void	getScreenShot(const char*	filename)
{
	glReadBuffer(GL_BACK);
	glReadPixels(0, 0, g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, frameBuffer);
	save_png(filename, frameBuffer, g_width, g_height, 4);
}

void	display()
{
	if(wiredFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glClearColor(0,0,0,0);

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	if(g_settings.useGraphics)
	{

		glLoadIdentity();

		glColor3f(1,0,0);

		glTranslatef(position.x,position.y, position.z);

		glRotatef(rotation.x,1,0,0);
		glRotatef(rotation.y,0,1,0);

		solver->RenderModels();

		DrawBBoxColored(g_settings.boundaries);

		glPointSize(6);

		pointSprite.Bind(0);

		glEnable(GL_POINT_SPRITE);
		glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

		glEnable(GL_BLEND);

	//	glDisable(GL_DEPTH_TEST);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		solver->Render();

		glDisable(GL_BLEND);

		pointSprite.Unbind();

		glDisable(GL_POINT_SPRITE);
	}else
		pointSprite.Unbind();

	glLoadIdentity();

	glColor3f(1,0,0);

	glDisable(GL_DEPTH_TEST);

	setup2DView(800,600);

	DrawText2D(10, 10, "Press left mouse button and drag to move");

	DrawText2D(300, 10, "Complex Systems Simulation Lab");

	if(g_simState.paused)
		DrawText2D(300, 20, "Simulation is paused");

	DrawText2D(10, 20, "Press right mouse button and drag to rotate");
	DrawText2D(10, 30, "Press +/- to zoom");
	DrawText2D(10, 40, "Press space to toggle pause");

	static	char s[100];
	sprintf(s,"FPS: %4.2f",g_simState.fps);

	DrawText2D(10, 50, s);

	sprintf(s,"Simulation time: %f", g_simState.simTime);

	DrawText2D(10, 60, s);

	sprintf(s,"Real time: %4.2f", g_simState.realTime);

	DrawText2D(10, 70, s);

	reshape(g_width, g_height);

	glEnable(GL_DEPTH_TEST);

	if(g_webService && takeScreenshot)
	{
		glFinish();
		getScreenShot(screenShotName.c_str());
		takeScreenshot = false;
	}
	else
		glFlush();

	glutSwapBuffers();
}

void	mouse(int btn, int state, int x, int y)
{
	if(g_webService)
		return;

	if(state == GLUT_DOWN)
	{
		if(btn == GLUT_LEFT_BUTTON)
			g_mouse_state.leftBtnPressed = true;
		if(btn == GLUT_RIGHT_BUTTON)
			g_mouse_state.rightBtnPressed = true;

		g_mouse_state.pos_x = x;
		g_mouse_state.pos_y = y;
	}else
	{
		if(btn == GLUT_LEFT_BUTTON)
			g_mouse_state.leftBtnPressed = false;
		if(btn == GLUT_RIGHT_BUTTON)
			g_mouse_state.rightBtnPressed = false;

		g_mouse_state.pos_x = -1;
		g_mouse_state.pos_y = -1;
	}

	if(btn == 3)
		position.z += 0.5f;

	if(btn == 4)
		position.z -= 0.5f;
}

void mouse_motion(int x,int y)
{
	if(g_webService)
		return;

	if(g_mouse_state.leftBtnPressed)
	{
		position.x += (x-g_mouse_state.pos_x)/100.0;
		position.y += (g_mouse_state.pos_y-y)/100.0;
	}

	if(g_mouse_state.rightBtnPressed)
	{
		rotation.x += (g_mouse_state.pos_y-y)/10.0;
		rotation.y += (x-g_mouse_state.pos_x)/10.0;
	}

	g_mouse_state.pos_x = x;
	g_mouse_state.pos_y = y;
}

void	update()
{
	static uint prevTime = glutGet(GLUT_ELAPSED_TIME);
	static float alpha = 0.0f;
	static bool	stopZoom = false;
	static uint frame = 0;
	static	float tt = 0.0f;

	static float startSimTime = 0;

	uint	time2 = glutGet(GLUT_ELAPSED_TIME);

	float	delta = (time2 - prevTime)/1000.0;
	frame++;

	tt += delta;

	if (tt >= 1.0f)
	{
			g_simState.fps = frame/tt;
			tt = 0.0f;
			frame = 0;
	}

	prevTime = time2;

	if(!g_simState.paused)
	{
		g_simState.advanceRealTime(delta);
		solver->Update(delta);
		g_simState.advanceSimTime(g_settings.timeStep);
	}

	if(g_webService)
	{
		if((startSimTime >= g_interval || g_simState.simTime-g_settings.startTime == 2*g_settings.timeStep
				|| g_simState.simTime >= g_settings.endTime) && g_interval != -1)
		{
			startSimTime = 0;
			char	buf[255];
			sprintf(buf,"screen_%f.png", g_simState.simTime);

//			std::string	fname = FilesPath::get().getFilePath(buf);

			screenShotName = FilesPath::get().getFilePath(buf);

			takeScreenshot = true;

//			getScreenShot(fname.c_str());
		}

		startSimTime += g_settings.timeStep;
	}

	if(g_simState.simTime > g_settings.endTime+g_settings.timeStep)
		throw	std::runtime_error("Simulation time is over");

	glutPostRedisplay();
}

void keyboard( unsigned char key, int x, int y) 
{
	if(g_webService)
		return;

    if(key == 27)
		throw	std::runtime_error("User pressed ESC");

    if(key == ' ')
    	g_simState.togglePause();
    if(key == '+')
    	position.z++;
    if(key == '-')
    	position.z--;

    if(key == 'w')
    	wiredFrame = !wiredFrame;

    if(key == 'n')
    	solver->ToggleRenderNormals();

    if(key == 'g')
    	solver->ToggleRenderGrid();
}

int main(int argc, char **argv)
{
	g_webService = false;

	std::ofstream	out_file;

	try
	{

	int devid = 0;

	if(argc < 2){
		std::cerr<<"configuration file name missing\n";
		return EXIT_FAILURE;
	}

	if(argc > 2 && std::string(argv[2]) == "-web")
	{
		g_webService = true;
		printf("Running in a WebService mode\n");

		if(argc > 3)
			g_interval = atoi(argv[3]);

		if(argc > 4)
			g_prefix = argv[4];
	}

	FilesPath::create((g_webService)?g_path:"", (g_prefix)?g_prefix:"");

	if(g_webService)
	{
		out_file.open(FilesPath::get().getFilePath("output.txt").c_str());
		LogFile::create(out_file);
	}else
		LogFile::create(std::cout);

	SmoldynParser	parser;
	parser(argv[1], g_settings);

	g_settings.calcParameters();

	LogFile::get()<<"Data is loaded. Dumping...\n";

	g_settings.dump(LogFile::get().get_stream());

	LogFile::get().get_stream().flush();

	position.x = -(g_settings.boundaries.max.x + g_settings.boundaries.min.x)/2;
	position.y = -(g_settings.boundaries.max.y + g_settings.boundaries.min.y)/2;
	position.z = -4*(g_settings.boundaries.max.z - g_settings.boundaries.min.z)/2;

	g_simState.simTime = g_settings.startTime;
	g_simState.paused =  (g_webService == false);

	cudaGLSetGLDevice(devid);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devid);

	LogFile::get()<<"Using device: "<<deviceProp.name<<"\n";

	g_width = 1024;
	g_height = 768;

	frameBuffer = new unsigned char[g_width*g_height*4];

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(g_width,g_height);
	glutCreateWindow("SmoldynGPU");
	
	glutDisplayFunc(display);
	glutIdleFunc(update);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(mouse_motion);

	glewInit();

	solver = new SmoldynSolver;

	solver->Init(g_settings);

	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_NORMALIZE);

	glEnable(GL_TEXTURE_2D);

	glShadeModel (GL_SMOOTH);

	glFrontFace(GL_CCW);

	glCullFace(GL_BACK);

	try{

		if(pointSprite.Load("sprite2.png"))
			pointSprite.Upload();
		else{
			std::cerr<<"Error loading point sprite\n";
			return EXIT_FAILURE;
		}
	}catch(...){
		std::cerr<<"Error loading point sprite\n";
		return EXIT_FAILURE;
	}

	glutMainLoop();
	std::cerr<<"here\n";

	}
	catch(std::exception& e)
	{
		std::cerr<<"Exception: "<<e.what()<<"\n";
	}

	LogFile::get()<<"Simulation run for "<<g_simState.realTime<<" seconds\n";

	delete solver;

	delete [] frameBuffer;

	LogFile::destroy();

	return 0;
}
