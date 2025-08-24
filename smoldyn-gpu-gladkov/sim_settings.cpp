/*
 * sim-settings.cpp
 *
 *  Created on: May 25, 2010
 *      Author: denis
 */

#include "sim_settings.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RapidXML/rapidxml.hpp"

#include "dsmc.h"

using namespace rapidxml;

settings_t::settings_t(): benchmark(false), cpu(false), geometry(true), mt(false), numParticles(SAMPLES_COUNT), maxRunTime(-1.0f),
							density(DSMC_DENSITY), fnum(DSMC_FNUM), temp(DSMC_T), grid_dim(GRID_DIM_X,GRID_DIM_Y,GRID_DIM_Z), dt(DSMC_DT),
								streamVelocity(0,0,0), periodicCondition('n'), viscValue(0.75f), csv(false), shaders(true)
{
	boundaries.reset(vec3_t(gridBoundaries_x,gridBoundaries_y,gridBoundaries_zMin),
						vec3_t(gridBoundaries_z,gridBoundaries_w,gridBoundaries_zMax));

	cell_size.reset(boundaries.get_xsize()/grid_dim.x,
						boundaries.get_ysize()/grid_dim.y,
						boundaries.get_zsize()/grid_dim.z);
}
//size_t fread ( void * ptr, size_t size, size_t count, FILE * stream );
void	settings_t::readFromFile(const char* filename)
{
	xml_document<> doc;

	FILE*	f = fopen(filename, "r");
	long size = ftell(f);

	fseek(f,0,SEEK_END);

	size = ftell(f) + 1;

	char*	text = new char[size];

	fseek(f,0,SEEK_SET);

	fread(text,size,1,f);

	fclose(f);

	text[size-1] = 0;

	doc.parse<0>(text);    // 0 means default parse flags

	xml_node<> *root = doc.first_node();

	xml_node<> *grid = root->first_node("grid");

	sscanf(grid->first_node("min")->value(), "%f %f %f", &boundaries.min.x, &boundaries.min.y, &boundaries.min.z);
	sscanf(grid->first_node("max")->value(), "%f %f %f", &boundaries.max.x, &boundaries.max.y, &boundaries.max.z);
	sscanf(grid->first_node("dim")->value(), "%d %d %d", &grid_dim.x, &grid_dim.y, &grid_dim.z);

	cell_size.reset(boundaries.get_xsize()/grid_dim.x,
						boundaries.get_ysize()/grid_dim.y,
						boundaries.get_zsize()/grid_dim.z);

	sscanf(root->first_node("density")->value(), "%f", &density);
	sscanf(root->first_node("fnum")->value(), "%f", &fnum);
	sscanf(root->first_node("temp")->value(), "%f", &temp);

	char*	svel = root->first_node("streamvel")->value();

	if(svel)
		sscanf(svel, "%f %f %f", &streamVelocity.x, &streamVelocity.y, &streamVelocity.z);

	if(root->first_node("periodic"))
		periodicCondition = root->first_node("periodic")->value()[0];

	if(root->first_node("model"))
		modelName = root->first_node("model")->value();

	if(root->first_node("statistics"))
	{
		statName = root->first_node("statistics")->value();
		if(root->first_node("statistics")->first_attribute("csv"))
			if(!strcmp(root->first_node("statistics")->first_attribute("csv")->value(),"true"))
				csv = true;
	}

	if(root->first_node("concentration"))
		concName = root->first_node("concentration")->value();

	if(root->first_node("velfield"))
		velsName = root->first_node("velfield")->value();

	std::string	gas_file;

	gas_file = root->first_node("gas")->first_attribute("file")->value();
	gas_name = root->first_node("gas")->value();

	delete [] text;

	f = fopen(gas_file.c_str(), "r");

	size = ftell(f);
	fseek(f,0,SEEK_END);
	size = ftell(f) + 1;
	text = new char[size];
	fseek(f,0,SEEK_SET);
	fread(text,size,1,f);
	fclose(f);
	text[size-1] = 0;

	xml_document<> gas_xml;    // character type defaults to char

	gas_xml.parse<0>(text);    // 0 means default parse flags

	xml_node<> *gas_root = gas_xml.first_node();

	xml_node<> *gnode = gas_root->first_node("gas");

	while(gas_name != gnode->first_attribute("name")->value())
	{
		gnode = gnode->next_sibling();
	}

	if(gnode)
	{
		sscanf(gnode->first_node("mass")->value(), "%f", &gas.molmass);
		sscanf(gnode->first_node("diam")->value(), "%f", &gas.diameter);
		sscanf(gnode->first_node("temp")->value(), "%f", &gas.Tref);
	}

	delete [] text;
}

void	settings_t::readFromCmdLine(int argc, char** argv)
{
	for(int i = 1; i < argc; i++)
	{
		if(!strcmp(argv[i],"-f"))
		{
			readFromFile(argv[i+1]);
			continue;
		}

		if(!strcmp(argv[i],"-benchmark"))
		{
			benchmark = true;
			continue;
		}

		if(!strcmp(argv[i],"-cpu"))
		{
			cpu = true;
			continue;
		}

		if(!strcmp(argv[i],"-no-geometry"))
		{
			geometry = false;
			continue;
		}

		if(!strcmp(argv[i],"-mt"))
		{
			mt = true;
			continue;
		}

		if(!strcmp(argv[i],"-no-shaders"))
		{
			shaders = false;
			continue;
		}

		if(!strcmp(argv[i],"-density"))
		{
			density = atof(argv[i+1]);
			continue;
		}

		if(!strcmp(argv[i],"-dt"))
		{
			dt = atof(argv[i+1]);
			continue;
		}

		if(!strcmp(argv[i],"-csv"))
		{
			csv = true;
			continue;
		}

		if(!strcmp(argv[i],"-run-for"))
		{
			maxRunTime = atof(argv[i+1]);
			continue;
		}
	}
}

void	settings_t::print()
{
	printf("Samples count: %d\n"
			"Benchmark mode is: %s\n"
			"CPU simulation is: %s\n"
			"Geometry is: %s\n"
			"Mersenne Twister is: %s\n", numParticles,benchmark?"ON":"OFF",cpu?"ON":"OFF",geometry?"ON":"OFF",mt?"ON":"OFF");

	printf("Grid configuration: <%f %f %f> to <%f %f %f> by <%d %d %d>\n",
			boundaries.min.x, boundaries.min.y, boundaries.min.z,
			boundaries.max.x, boundaries.max.y, boundaries.max.z,
			grid_dim.x, grid_dim.y, grid_dim.z);

	printf("Density %e\nFNum %e\nTemperature: %f\n", density, fnum, temp);

	printf("Gas properties\nName: %s\nMass: %e\nDiameter: %e\nAt temperature: %f\n", gas_name.c_str(), gas.molmass, gas.diameter, gas.Tref);
	printf("Time step is: %e\n", dt);

	printf("Stream velocity is: %f %f %f\n", streamVelocity.x, streamVelocity.y, streamVelocity.z);

	printf("Model name is: %s\n", modelName.c_str());
}
