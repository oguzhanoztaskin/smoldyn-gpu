/*
 * shader.cpp
 *
 *  Created on: Sep 30, 2010
 *      Author: denis
 */

#include "shader.h"

#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <iostream>

char *textFileRead(const char *fn) {
  FILE *fp;
  char *content = NULL;

  int count = 0;

  if (fn != NULL) {
    fp = fopen(fn, "rt");

    if (fp != NULL) {
      fseek(fp, 0, SEEK_END);
      count = ftell(fp);
      rewind(fp);

      if (count > 0) {
        content = (char *)malloc(sizeof(char) * (count + 1));
        count = fread(content, sizeof(char), count, fp);
        content[count] = '\0';
      }
      fclose(fp);
    }
  }

  return content;
}

void printShaderInfoLog(GLuint obj) {
  std::cout << "Shader info log\n";
  int infologLength = 0;
  int charsWritten = 0;
  char *infoLog;
  glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
  if (infologLength > 1) {
    infoLog = (char *)malloc(infologLength);
    glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);

    std::cout << "Shader error: \n\n" << infoLog << "\n";

    free(infoLog);
  }
}

void printProgramInfoLog(GLuint obj) {
  std::cout << "Program info log\n";
  int infologLength = 0;
  int charsWritten = 0;
  char *infoLog;
  glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
  if (infologLength > 1) {
    infoLog = (char *)malloc(infologLength);
    glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);

    std::cout << "Shader error: \n\n" << infoLog << "\n";
    free(infoLog);
  }
}

struct Shader::impl_t {
  GLuint program;
};

Shader::Shader() : m_pImpl(new impl_t) {}

Shader::~Shader() { delete m_pImpl; }

bool Shader::LoadAndCreateProgram(const char *vert, const char *geom,
                                  const char *frag) {
  assert(glGetError() == GL_NO_ERROR);

  char *gs = textFileRead(geom);
  if (gs == NULL) {
    std::cout << "Cannot load geometry shader\n";
    return false;
  }

  GLuint g = glCreateShader(GL_GEOMETRY_SHADER);

  const char *ss = gs;
  glShaderSource(g, 1, (const GLcharARB **)&ss, NULL);
  glCompileShader(g);
  GLint res;
  glGetShaderiv(g, GL_COMPILE_STATUS, &res);

  free(gs);

  if (res != GL_TRUE) {
    printShaderInfoLog(g);
    return false;
  }

  char *fs = textFileRead(frag);
  if (fs == NULL) {
    std::cout << "Cannot load fragment shader\n";
    return false;
  }

  GLuint f = glCreateShader(GL_FRAGMENT_SHADER);

  const char *ff = fs;
  glShaderSource(f, 1, (const GLcharARB **)&ff, NULL);
  glCompileShader(f);

  glGetShaderiv(f, GL_COMPILE_STATUS, &res);

  free(fs);

  if (res != GL_TRUE) {
    printShaderInfoLog(f);
    return false;
  }

  char *vs = textFileRead(vert);
  if (vs == NULL) {
    std::cout << "Cannot load vertex shader\n";
    return false;
  }

  GLuint v = glCreateShader(GL_VERTEX_SHADER);

  const char *vv = vs;
  glShaderSource(v, 1, (const GLcharARB **)&vv, NULL);
  glCompileShader(v);
  glGetShaderiv(v, GL_COMPILE_STATUS, &res);

  free(vs);

  if (res != GL_TRUE) {
    printShaderInfoLog(f);
    return false;
  }

  m_pImpl->program = glCreateProgram();

  glAttachShader(m_pImpl->program, v);
  glAttachShader(m_pImpl->program, f);
  glAttachShader(m_pImpl->program, g);

  return true;
}

void Shader::SetParameter(GLenum en, GLint val) {
  glProgramParameteriARB(m_pImpl->program, en, val);
  assert(glGetError() == GL_NO_ERROR);
}

bool Shader::CompileAndLink() {
  glLinkProgram(m_pImpl->program);
  assert(glGetError() == GL_NO_ERROR);

  GLint res = 0;

  glGetProgramiv(m_pImpl->program, GL_LINK_STATUS, &res);
  assert(glGetError() == GL_NO_ERROR);

  if (res != GL_TRUE) {
    printProgramInfoLog(m_pImpl->program);
    return false;
  }

  return true;
}

void Shader::Bind() {
  glUseProgram(m_pImpl->program);
  assert(glGetError() == GL_NO_ERROR);
}

void Shader::Unbind() { glUseProgram(0); }
