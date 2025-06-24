#pragma once
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glad.h>
#include <cuda_gl_interop.h>
#include "shader.h"
#include "CudaTexture.cuh"

constexpr float PI = 3.14159265359f;
constexpr float TAU = 6.28318530718f;
constexpr float PD2 = 1.57079632679f;

template<size_t activeSize>
class NetworkCudaTexture {

	const float nodePositionMult() const { return 2.0f / ((float)activeSize - 1.0f); };
	const float nodeIndexMult() const { return 1.0f / (float)activeSize; };
	const float nodeRadiusMult() const { return PD2 / (float)activeSize; };
	const float minDrawRadius() const {
		const float s = 1.0f / std::cos(nodeRadiusMult());
		const float t = std::tan(nodeRadiusMult());
		return (s - t) / (s + t);
	};
	const float maxDrawRadius() const { return 1.0f; };

	float energy;
	float4 aabb;

	CudaTexture values, errors, weightsE, weightsV;

	float* valuesN;
	float* errorsN;

	inline void calcNorm(float* input, float* output);
	inline void calcNormAF(float* input,float* output);
	inline void calcValues();
	inline void calcErrors();
	inline void calcWeights();

public:

	NetworkCudaTexture();
	~NetworkCudaTexture();

	void reset();
	void resetValues();
	void resetNValues();
	void resetErrors();
	void resetNErrors();
	void resetMatrixValues();
	void resetMatrixErrors();

	void CudaEnableValues();
	void CudaEnableErrors();
	void CudaEnableWeights();

	void CudaDisableValues();
	void CudaDisableErrors();
	void CudaDisableWeights();

	void train(float* input, float* output, uint32_t* inputIDs, uint32_t* outputIDs, uint32_t sizeI, uint32_t sizeO);
	void run(float* input, uint32_t* inputIDs, uint32_t sizeI);
	void sleep();

	void draw();
	float getEnergy() const { return energy; };
	//void testMatrixMultiplication();

	//void testVectorNormalization();

	void print() {
		std::cout << "Energy: " << energy << '\n';
	}

};

template class NetworkCudaTexture<64u>;