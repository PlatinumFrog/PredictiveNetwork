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

template<size_t activeSize>
class NetworkCuda {

	float* values;
	float* valuesN;
	float* errors;
	float* errorsN;
	float* weightsV;
	float* weightsE;

public:

	NetworkCuda();
	~NetworkCuda();
	
	const size_t const getActiveSize() const { return activeSize; };
	const size_t const getWeightSize() const { return activeSize * activeSize; };
	const size_t const getBlockSize() const { return 1024ull; };
	const size_t const getGridSize() const { return (activeSize / getBlockSize()) + 1ull; };

	float run();
	
	//void testMatrixMultiplication();
};

template class NetworkCuda<4096u>;