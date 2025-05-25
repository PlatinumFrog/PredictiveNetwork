#include "NetworkCuda.cuh"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define GPU_ERROR_RET(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; return false; }
#define GPU_ERROR_ABT(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; abort(); }
__global__ void updateVector(
	const float* vector, 
	const float* matrix,
	float* output,
	const size_t sizeV, 
	const size_t sizeM,
	const size_t sizeO
) {

	__shared__ float sums[1024u];

	sums[threadIdx.x] = 0.0f;

	//Global IDs
	uint32_t vectorID1 = threadIdx.x;
	uint32_t vectorID2 = vectorID1 + blockDim.x;
	uint32_t matrixID1 = sizeV * blockIdx.x + threadIdx.x;
	uint32_t matrixID2 = matrixID1 + blockDim.x;

	while (vectorID1 < sizeV && vectorID2 < sizeV && matrixID1 < sizeM && matrixID2 < sizeM) {
		sums[threadIdx.x] += vector[vectorID1] * matrix[matrixID1] + vector[vectorID2] * matrix[matrixID2];
		vectorID1 += blockDim.x;
		matrixID1 += blockDim.x;
		vectorID2 += blockDim.x;
		matrixID2 += blockDim.x;
	}
	if (threadIdx.x < 512u) sums[threadIdx.x] += sums[threadIdx.x + 512u];
	if (threadIdx.x < 256u) sums[threadIdx.x] += sums[threadIdx.x + 256u];
	if (threadIdx.x < 128u) sums[threadIdx.x] += sums[threadIdx.x + 128u];
	if (threadIdx.x < 64u) sums[threadIdx.x] += sums[threadIdx.x + 64u];
	if (threadIdx.x < 32u) sums[threadIdx.x] += sums[threadIdx.x + 32u];
	if (threadIdx.x < 16u) sums[threadIdx.x] += sums[threadIdx.x + 16u];
	if (threadIdx.x < 8u) sums[threadIdx.x] += sums[threadIdx.x + 8u];
	if (threadIdx.x < 4u) sums[threadIdx.x] += sums[threadIdx.x + 4u];
	if (threadIdx.x < 2u) sums[threadIdx.x] += sums[threadIdx.x + 2u];
	if (threadIdx.x < 1u) sums[threadIdx.x] += sums[threadIdx.x + 1u];

	output[blockIdx.x] = sums[0u];
}

NetworkCuda::NetworkCuda():
values(nullptr),
errors(nullptr),
weightsV(nullptr),
weightsE(nullptr)
{
	/*cudaMalloc((void**)&values, getVectorSize() * sizeof(float));
	cudaMalloc((void**)&errors, getVectorSize() * sizeof(float));
	cudaMalloc((void**)&weightsV, getMatrixSize() * sizeof(float));
	cudaMalloc((void**)&weightsE, getMatrixSize() * sizeof(float));*/
};

NetworkCuda::~NetworkCuda() {
	/*cudaFree(values);
	cudaFree(errors);
	cudaFree(weightsV);
	cudaFree(weightsE);*/
};

void NetworkCuda::testMatrixMultiplication() {
	const static uint32_t sizesV = 4096u;
	const static uint32_t sizesM = sizesV * sizesV;

	float* testVector_d = nullptr;
	float* testMatrix_d = nullptr;
	float* testResults_d = nullptr;

	float* testVector_h = nullptr;
	float* testMatrix_h = nullptr;
	float* testResults_h = nullptr;

	float* testResults_dh = nullptr;

	testVector_h = new float[sizesV];
	testMatrix_h = new float[sizesM];
	testResults_h = new float[sizesV];
	testResults_dh = new float[sizesV];
	for (uint32_t u = 0u; u < sizesV; u++) {
		testVector_h[u] = ((float)std::rand() / (float)RAND_MAX);
		testResults_h[u] = 0u;
		for (uint32_t v = 0u; v < sizesV; v++) {
			testMatrix_h[v + (sizesV * u)] = ((float)std::rand() / (float)RAND_MAX);
		}
	}

	cudaError_t err;
	err = cudaMalloc((void**)&testVector_d, sizesV * sizeof(float));
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaMalloc((void**)&testMatrix_d, sizesM * sizeof(float));
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaMalloc((void**)&testResults_d, sizesV * sizeof(float));
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaMemcpy(testVector_d, testVector_h, sizesV * sizeof(float), cudaMemcpyHostToDevice);
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaMemcpy(testMatrix_d, testMatrix_h, sizesM * sizeof(float), cudaMemcpyHostToDevice);
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	for (uint32_t u = 0u; u < sizesV; u++) {
		for (uint32_t v = 0u; v < sizesV; v++) {
			testResults_h[u] += testVector_h[v] * testMatrix_h[(sizesV * u) + v];
		}
	}

	delete[] testVector_h;
	delete[] testMatrix_h;

	updateVector<<<1024u,1024u>>>(testVector_d, testMatrix_d, testResults_d, sizesV, sizesM, sizesV);
	cudaDeviceSynchronize();
	err = cudaMemcpy(testResults_dh, testResults_d, sizesV * sizeof(float), cudaMemcpyDeviceToHost);
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaFree(testVector_d);
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaFree(testMatrix_d);
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	err = cudaFree(testResults_d);
	GPU_ERROR_ABT("Error: ", err);
	cudaDeviceSynchronize();
	float score = 0.0f;
	for (uint32_t u = 0u; u < sizesV; u++) {
		float sum = testResults_h[u] - testResults_dh[u];
		score += sum * sum;
	}

	delete[] testResults_h;
	delete[] testResults_dh;
	
	std::cout << "Matrix Kernel Test Results: " << score << '\n';
};