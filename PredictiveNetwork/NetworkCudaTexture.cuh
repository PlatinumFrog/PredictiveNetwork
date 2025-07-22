#pragma once
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glad.h>
#include <cuda_gl_interop.h>
#include "shader.h"
#include "CudaTexture.cuh"

template<size_t activeSize>
class NetworkCudaTexture {

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
	NetworkCudaTexture(NetworkCudaTexture& n);
	NetworkCudaTexture(float4 ab);
	~NetworkCudaTexture();

	void zero();
	void zeroValues();
	void zeroNValues();
	void zeroErrors();
	void zeroNErrors();
	void zeroMatrixValues();
	void zeroMatrixErrors();

	void randomize();
	void randomizeValues();
	void randomizeErrors();
	void randomizeMatrixValues();
	void randomizeMatrixErrors();

	void displace();
	void displaceValues();
	void displaceErrors();
	void displaceMatrixValues();
	void displaceMatrixErrors();

	void enableCuda();
	void enableCudaValues();
	void enableCudaErrors();
	void enableCudaWeights();

	void disableCuda();
	void disableCudaValues();
	void disableCudaErrors();
	void disableCudaWeights();

	float4 getAABB() const;
	void setAABB(float x, float y, float width, float height);
	void setAABB(float4 ab);

	void setValues(uint32_t index, uint32_t size, float* data);
	void getValues(uint32_t index, uint32_t size, float* data);

	void setErrors(uint32_t index, uint32_t size, float* data);
	void getErrors(uint32_t index, uint32_t size, float* data);

	void train();

	void draw();
	float getEnergy() const { return energy; };
	//void testMatrixMultiplication();
	//void testVectorNormalization();

	void print() {
		std::cout << "Energy: " << energy << '\n';
	}

};

template class NetworkCudaTexture<1024u>;