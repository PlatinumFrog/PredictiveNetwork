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

//TODO: Add layer IO kernels

template<size_t activeSize>
class NetworkCuda {

	float* values;
	float* valuesN;
	float* errors;
	float* errorsN;
	float* weightsV;
	float* weightsE;

	inline void calcNorm(float* input, float* output);
	inline void calcNormAF(float* input,float* output);
	inline void calcValues();
	inline void calcErrors();
	inline void calcWeights();

public:

	NetworkCuda();
	~NetworkCuda();
	
	const size_t const getActiveSize() const { return activeSize; };
	const size_t const getWeightSize() const { return activeSize * activeSize; };
	const size_t const getBlockSize() const { return 1024ull; };
	const size_t const getGridSize() const { return (activeSize / getBlockSize()) + 1ull; };

	void reset();
	void resetValues();
	void resetNValues();
	void resetErrors();
	void resetNErrors();
	void resetMatrixValues();
	void resetMatrixErrors();

	float train(float* input, float* output, uint32_t* inputIDs, uint32_t* outputIDs, uint32_t sizeI, uint32_t sizeO);
	
	//void testMatrixMultiplication();

	//void testVectorNormalization();
};

template class NetworkCuda<64u>;