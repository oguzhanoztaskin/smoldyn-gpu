/*
 * shader.h
 *
 *  Created on: Sep 30, 2010
 *      Author: denis
 */

#ifndef SHADER_H_
#define SHADER_H_

#include <GL/glew.h>

//The very debug implementation of shader program
/*
 * 	glProgramParameteri(p,GL_GEOMETRY_INPUT_TYPE_EXT,GL_TRIANGLES);
	glProgramParameteri(p,GL_GEOMETRY_OUTPUT_TYPE_EXT,GL_TRIANGLE_STRIP);
	assert(glGetError()==GL_NO_ERROR);

	int temp;
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&temp);
	glProgramParameteri(p,GL_GEOMETRY_VERTICES_OUT_EXT,temp);
	assert(glGetError()==GL_NO_ERROR);
 *
 */
class	Shader
{
public:
	Shader();
	~Shader();
	bool	LoadAndCreateProgram(const char* vert, const char*	geom, const char* frag);
	bool	CompileAndLink();
	void	SetParameter(GLenum, GLint);
	void	Bind();
	void	Unbind();
private:
	struct	impl_t;
	impl_t*	m_pImpl;
};

#endif /* SHADER_H_ */
