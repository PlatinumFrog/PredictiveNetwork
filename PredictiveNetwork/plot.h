#pragma once
#ifdef _DEBUG
#include "gladDebug/include/glad.h"
#else
#include "gladRelease/include/glad.h"
#endif
#include "shader.h"
#include "matrix.cuh"
#include "vector.cuh"
#include <vector>

constexpr uint32_t maxBufferLength = 4096;

class Plot {
	GLuint vboID, vaoID, shaderID;
	uint32_t length;
	float4 box;
	float4 color;
	float2* data;
	
public:
	Plot():
		vboID(0),
		vaoID(0),
		shaderID(0),
		box{-1.0f, -1.0f, 1.0f, 1.0f},
		color(1.0f, 1.0f, 1.0f, 1.0f),
		data(new float2[maxBufferLength])
	{
		for (uint32_t i = 0; i < maxBufferLength; i++) data[i] = float2{0.0f, 0.0f};

		glGenBuffers(1, &vboID);
		glGenVertexArrays(1, &vaoID);
		
		glBindVertexArray(vaoID);
		glBindBuffer(GL_ARRAY_BUFFER, vboID);
		glBufferData(GL_ARRAY_BUFFER, maxBufferLength * sizeof(float2), &data, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

	};
	
	void setColor(float greyScale);
	void setColor(float r, float g, float b, float a);
	void setColor(float4 c);
	
	void addDataPoint(float x, float y);
	void addDataPoint(float2 p);

};