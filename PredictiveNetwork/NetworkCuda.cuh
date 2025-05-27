#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glad.h>
#include <cuda_gl_interop.h>
#include "shader.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>

constexpr float PI = 3.14159265359f;
constexpr float TAU = 6.28318530718f;
constexpr float PD2 = 1.57079632679f;

template<size_t activeSize>
class NetworkCuda {

	const float nodePositionMult() { return 2.0f / ((float)activeSize - 1.0f); };
	const float nodeIndexMult() { return 1.0f / (float)activeSize; };
	const float nodeRadiusMult() { return PD2 / (float)activeSize; };
	const float minDrawRadius() {
		const float s = 1.0f / std::cos(nodeRadiusMult());
		const float t = std::tan(nodeRadiusMult());
		return (s - t) / (s + t);
	};
	const float maxDrawRadius() { return 1.0f; };

	uint32_t flags;

	GLuint
		nodePosVBO, nodeValVBO, nodeErrVBO, nodeVAO;
	GLuint nodeShader;

		//weightPosVBO, weightCol1VBO, weightCol2VBO, weightVAO, weightShader;
	cudaGraphicsResource* cgrValues;
	cudaGraphicsResource* cgrErrors;
	cudaGraphicsResource* cgrNodePos;
	//cudaGraphicsResource* cgrWeights;

	float* values;
	float* valuesN;
	float* errors;
	float* errorsN;
	float* weightsV;
	float* weightsE;

	float* nodePos;

	inline void calcNorm(float* input, float* output);
	inline void calcNormAF(float* input,float* output);
	inline void calcValues();
	inline void calcErrors();
	inline void calcWeights();

public:

	NetworkCuda();
	~NetworkCuda();

	void reset();
	void resetValues();
	void resetNValues();
	void resetErrors();
	void resetNErrors();
	void resetMatrixValues();
	void resetMatrixErrors();
	void resetPositions();

	void CudaEnableValues();
	void CudaEnableErrors();
	void CudaEnableNodePos();

	void CudaDisableValues();
	void CudaDisableErrors();
	void CudaDisableNodePos();

	float train(float* input, float* output, uint32_t* inputIDs, uint32_t* outputIDs, uint32_t sizeI, uint32_t sizeO);
	
	void draw();

	//void testMatrixMultiplication();

	//void testVectorNormalization();
};

template class NetworkCuda<4096u>;