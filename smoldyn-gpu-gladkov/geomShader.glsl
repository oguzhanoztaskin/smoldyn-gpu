#version 120
#extension GL_EXT_geometry_shader4 : require
#extension GL_EXT_gpu_shader4 : require

#define	TRI_SIZE 0.05f

void main() 
{
	gl_FrontColor=gl_FrontColorIn[0];

	vec4	vert1 = gl_PositionIn[0] + gl_ModelViewMatrix*vec4(0.0,-TRI_SIZE,TRI_SIZE,0.0);
	vec4	vert2 = gl_PositionIn[0] + gl_ModelViewMatrix*vec4(TRI_SIZE,-TRI_SIZE,-TRI_SIZE,0.0);
	vec4	vert3 = gl_PositionIn[0] + gl_ModelViewMatrix*vec4(-TRI_SIZE,-TRI_SIZE,-TRI_SIZE,0.0);
	vec4	vert4 = gl_PositionIn[0] + gl_ModelViewMatrix*vec4(0.0,TRI_SIZE,0.0,0.0);

	vec3	norm1 = normalize(gl_NormalMatrix*vec3(0.0,-TRI_SIZE,TRI_SIZE));
	vec3	norm2 = normalize(gl_NormalMatrix*vec3(TRI_SIZE,-TRI_SIZE,-TRI_SIZE));
	vec3	norm3 = normalize(gl_NormalMatrix*vec3(-TRI_SIZE,-TRI_SIZE,-TRI_SIZE));
	vec3	norm4 = normalize(gl_NormalMatrix*vec3(0.0,TRI_SIZE,0.0));

	vec3 lightDir = normalize(vec3(gl_LightSource[0].position));

	vec4	col1 = gl_FrontColorIn[0] * gl_LightSource[0].diffuse * max(dot(norm1, lightDir), 0.0);
	vec4	col2 = gl_FrontColorIn[0] * gl_LightSource[0].diffuse * max(dot(norm2, lightDir), 0.0);
	vec4	col3 = gl_FrontColorIn[0] * gl_LightSource[0].diffuse * max(dot(norm3, lightDir), 0.0);
	vec4	col4 = gl_FrontColorIn[0] * gl_LightSource[0].diffuse * max(dot(norm4, lightDir), 0.0);


	//bottom tri

	gl_Position = vert1;
	gl_FrontColor= col1;
	EmitVertex();

	gl_Position = vert2;
	gl_FrontColor= col2;
	EmitVertex();

	gl_Position = vert3;
	gl_FrontColor= col3;
	EmitVertex();

	EndPrimitive();

	//right tri

	gl_Position = vert1;
	gl_FrontColor= col1;
	EmitVertex();

	gl_Position = vert2;
	gl_FrontColor= col2;
	EmitVertex();

	gl_Position = vert4;
	gl_FrontColor= col4;
	EmitVertex();

	EndPrimitive();

	//back tri

	gl_Position = vert2;
	gl_FrontColor= col2;
	EmitVertex();

	gl_Position = vert4;
	gl_FrontColor= col4;
	EmitVertex();

	gl_Position = vert3;
	gl_FrontColor= col3;
	EmitVertex();

	EndPrimitive();

	//left tri

	gl_Position = vert3;
	gl_FrontColor= col3;
	EmitVertex();

	gl_Position = vert1;
	gl_FrontColor= col1;
	EmitVertex();

	gl_Position = vert4;
	gl_FrontColor= col4;
	EmitVertex();

	EndPrimitive();
}

