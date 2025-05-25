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
	uint32_t matrixID1 = sizeV * blockIdx.x + threadIdx.x;

	while (vectorID1 < sizeV && matrixID1 < sizeM) {
		sums[threadIdx.x] += vector[vectorID1] * matrix[matrixID1];
		vectorID1 += blockDim.x;
		matrixID1 += blockDim.x;
	}

	__syncthreads();
	if (threadIdx.x < 512u) sums[threadIdx.x] += sums[threadIdx.x + 512u];
	__syncthreads();
	if (threadIdx.x < 256u) sums[threadIdx.x] += sums[threadIdx.x + 256u];
	__syncthreads();
	if (threadIdx.x < 128u) sums[threadIdx.x] += sums[threadIdx.x + 128u];
	__syncthreads();
	if (threadIdx.x < 64u) sums[threadIdx.x] += sums[threadIdx.x + 64u];
	__syncthreads();
	if (threadIdx.x < 32u) sums[threadIdx.x] += sums[threadIdx.x + 32u];
	__syncthreads();
	if (threadIdx.x < 16u) sums[threadIdx.x] += sums[threadIdx.x + 16u];
	__syncthreads();
	if (threadIdx.x < 8u) sums[threadIdx.x] += sums[threadIdx.x + 8u];
	__syncthreads();
	if (threadIdx.x < 4u) sums[threadIdx.x] += sums[threadIdx.x + 4u];
	__syncthreads();
	if (threadIdx.x < 2u) sums[threadIdx.x] += sums[threadIdx.x + 2u];
	__syncthreads();
	if (threadIdx.x < 1u) sums[threadIdx.x] += sums[threadIdx.x + 1u];
	__syncthreads();
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
	const static uint32_t sizesV = 1024u;
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

	std::srand(334);

	for (uint32_t u = 0u; u < sizesV; u++) {
		testVector_h[u] = ((float)std::rand() / (float)RAND_MAX);
		testResults_h[u] = 0u;
		for (uint32_t v = 0u; v < sizesV; v++) {
			testMatrix_h[v + (sizesV * u)] = ((float)std::rand() / (float)RAND_MAX);
		}
	}
	cudaMalloc((void**)&testVector_d, sizesV * sizeof(float));
	cudaMalloc((void**)&testMatrix_d, sizesM * sizeof(float));
	cudaMalloc((void**)&testResults_d, sizesV * sizeof(float));
	cudaMemcpy(testVector_d, testVector_h, sizesV * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(testMatrix_d, testMatrix_h, sizesM * sizeof(float), cudaMemcpyHostToDevice);
	for (uint32_t u = 0u; u < sizesV; u++) {
		for (uint32_t v = 0u; v < sizesV; v++) {
			testResults_h[u] += testVector_h[v] * testMatrix_h[(sizesV * u) + v];
		}
	}

	delete[] testVector_h;
	delete[] testMatrix_h;

	updateVector<<<sizesV,(sizesV > 1024u) ? 1024u : sizesV>>>(testVector_d, testMatrix_d, testResults_d, sizesV, sizesM, sizesV);
	cudaDeviceSynchronize();
	cudaMemcpy(testResults_dh, testResults_d, sizesV * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(testVector_d);
	cudaFree(testMatrix_d);
	cudaFree(testResults_d);
	float score = 0.0f;
	for (uint32_t u = 0u; u < sizesV; u++) {
		float sum = testResults_h[u] - testResults_dh[u];
		score += sum * sum;
	}

	delete[] testResults_h;
	delete[] testResults_dh;
	
	std::cout << "Matrix Kernel Test Results: " << score << '\n';
};