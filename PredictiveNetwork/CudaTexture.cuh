#pragma once
#ifdef _DEBUG
#include "gladDebug/include/glad.h"
#else
#include "gladRelease/include/glad.h"
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "shader.h"
#include "matrix.cuh"
#include "vector.cuh"
#include <iostream>

#define GPU_ERROR_RET(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; return false; }
#define GPU_ERROR_ABT(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; abort(); }

class CudaTexture {
	GLuint pbo, textureID, vao, vbo;
	shader shaderProgram;
	cudaGraphicsResource* cudaPBO;
	matrix4 vertices;
	uint32_t width, height;

public:

	float3* data;
	bool cudaEnabled;
	bool isDrawable;

	CudaTexture():
		pbo(0),
		textureID(0),
		vao(0),
		vbo(0),
		shaderProgram(Shader::create("CudaTexture.vert", "CudaTexture.frag")),
		cudaPBO(nullptr),
		vertices{
			float4{ 1.0f,  1.0f, 0.0f, 0.0f},
			float4{ 1.0f, -1.0f, 0.0f, 1.0f},
			float4{-1.0f,  1.0f, 1.0f, 0.0f},
			float4{-1.0f, -1.0f, 1.0f, 1.0f}
		},
		width(0),
		height(0),
		cudaEnabled(false),
		data(nullptr),
		isDrawable(false) {
		// Initialize VAO and VBO
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &pbo);
		glGenTextures(1, &textureID);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

	}

	bool setRes(uint32_t w, uint32_t h) {
		if (w != width || h != height) {
			width = w;
			height = h;

			bool cudaWasEnabled = cudaEnabled;
			cudaError_t err;

			if (cudaPBO != nullptr) {
				if (cudaEnabled) if (!disableCuda()) return false;
				err = cudaGraphicsUnregisterResource(cudaPBO);
				GPU_ERROR_RET("Error unregistering cuda buffer", err);
				cudaPBO = nullptr;
			}

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glBindTexture(GL_TEXTURE_2D, 0);

			glDeleteBuffers(1, &pbo);
			glDeleteTextures(1, &textureID);
			glGenBuffers(1, &pbo);
			glGenTextures(1, &textureID);

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float3), nullptr, GL_DYNAMIC_COPY);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			glBindTexture(GL_TEXTURE_2D, textureID);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glBindTexture(GL_TEXTURE_2D, 0);

			err = cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsNone);
			GPU_ERROR_RET("Error registering cuda buffer", err);

			if (cudaWasEnabled) { if (!enableCuda()) return false; }
		}
		return true;
	}

	void setRect(float x, float y, float width, float height) {
		vertices = matrix4{
			float4{x, y, 1.0f, 1.0f},
			float4{x + width, y, 0.0f, 1.0f},
			float4{x, y + height, 1.0f, 0.0f},
			float4{x + width, y + height, 0.0f, 0.0f}
		};
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	bool enableCuda() {
		if (!cudaEnabled && cudaPBO) {
			cudaError_t err = cudaGraphicsMapResources(1, &cudaPBO, 0);
			GPU_ERROR_RET("Error mapping cuda buffer", err);
			size_t numBytes;
			err = cudaGraphicsResourceGetMappedPointer((void**)&data, &numBytes, cudaPBO);
			GPU_ERROR_RET("Error getting cuda buffer", err);
			cudaEnabled = true;
			isDrawable = false;
		}
		return true;
	}

	bool disableCuda() {
		if (cudaEnabled && cudaPBO) {
			cudaError_t err = cudaGraphicsUnmapResources(1, &cudaPBO, 0);
			GPU_ERROR_RET("Error unmapping cuda buffer", err);
			cudaEnabled = false;
			isDrawable = true;
			data = nullptr;
		}
		return true;
	}

	void updateTexture() const {
		if (!isDrawable) return;
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	void draw() const {
		if (!isDrawable) return;
		glUseProgram(shaderProgram.id);
		glBindVertexArray(vao);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
		glUseProgram(0);
	}

	uint32_t getWidth() const {
		return width;
	}

	uint32_t getHeight() const {
		return height;
	}

	static friend void operator<<(CudaTexture& t1, CudaTexture& t2) {
		t1.setRes(t2.width, t2.height);
		if (t1.cudaEnabled && t2.cudaEnabled) {
			cudaError_t err = cudaMemcpy(t1.data, t2.data, t1.width * t1.height * sizeof(float3), cudaMemcpyDeviceToDevice);
			GPU_ERROR_ABT("Error copying cuda buffer", err);
		} else {
			t1.enableCuda();
			t2.enableCuda();
			cudaError_t err = cudaMemcpy(t1.data, t2.data, t1.width * t1.height * sizeof(float3), cudaMemcpyDeviceToDevice);
			GPU_ERROR_ABT("Error copying cuda buffer", err);
			t1.disableCuda();
			t2.disableCuda();
		}
	}

};