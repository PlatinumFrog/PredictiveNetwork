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

class NetworkCuda {

	float* values;
	float* errors;
	float* weightsV;
	float* weightsE;

public:

	NetworkCuda();
	~NetworkCuda();

	/*inline const size_t getVectorSize() const { return activeSize; }
	inline const size_t getMatrixSize() const { return activeSize * activeSize; }

	*/void testMatrixMultiplication();
};