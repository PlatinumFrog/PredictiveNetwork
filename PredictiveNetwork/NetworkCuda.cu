#include "NetworkCuda.cuh"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define GPU_ERROR_RET(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; return false; }
#define GPU_ERROR_ABT(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; abort(); }

__device__ float AF(float x) {
	return x / (1.0f + std::abs(x));
}

__device__ float AFD(float x) {
	float i = (1.0f + std::abs(x));
	return 1.0f / (i * i);
}
//
//// Sum reduction from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//// Then converted to large vector magnitude calculation

template<size_t blockSize>
__global__ void sumReduceInitial(float* g_idata, float* g_odata, uint32_t n) {
	// Shared Memory for operation
	extern __shared__ float sdata[];

	// Shared Memory index
	uint32_t tid = threadIdx.x;

	// Global Memory index
	uint32_t i = blockIdx.x * blockSize + tid;

	// Total Grid Size
	uint32_t gridSize = blockSize * gridDim.x;

	// Set Shared Memory to zero
	sdata[tid] = 0.0f;

	// Keep adding blocks of memory until the end of the array
	while (i < n) {
		float x = AF(g_idata[i]);
		sdata[tid] += x * x;
		i += gridSize;
	}

	// Make sure that all thread independant shared memory operations are done
	__syncthreads();

	// Start tree based reduction
	if (blockSize >= 1024u) { if (tid < 512u) { sdata[tid] += sdata[tid + 512u]; } __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) { sdata[tid] += sdata[tid + 256u]; } __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; } __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; } __syncthreads(); }

	// Start warp level reduction
	if (tid < 32u) {
		volatile float* s = sdata;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}

	// Write result to first half of the global memory array
	if (tid == 0u) g_odata[blockIdx.x] = sdata[0];
}

template<size_t blockSize>
__global__ void sumReduceContinued(float* g_idata, float* g_odata, uint32_t n) {
	// Shared Memory for operation
	extern __shared__ float sdata[];

	// Shared Memory index
	uint32_t tid = threadIdx.x;

	// Global Memory index
	uint32_t i = blockIdx.x * blockSize + tid;

	// Total Grid Size
	uint32_t gridSize = blockSize * gridDim.x;

	// Set Shared Memory to zero
	sdata[tid] = 0.0f;

	// Keep adding grid sized blocks of memory until the end of the array
	while (i < n) {
		sdata[tid] += g_idata[i];
		i += gridSize;
	}

	// Make sure that all thread independant shared memory operations are done
	__syncthreads();

	// Start tree based reduction
	if (blockSize >= 1024u) { if (tid < 512u) { sdata[tid] += sdata[tid + 512u]; } __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) { sdata[tid] += sdata[tid + 256u]; } __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; } __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; } __syncthreads(); }

	// Start warp level reduction
	if (tid < 32u) {
		volatile float* s = sdata;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}

	// Write result to first half of the global memory array
	if (tid == 0u) g_odata[blockIdx.x] = sdata[0u];
}

template<size_t blockSize>
__global__ void updateValues(
	float* values,
	float* errors,
	float* matrix,
	const size_t sizeV
) {

	extern __shared__ float sums[];

	uint32_t tid = threadIdx.x;

	sums[tid] = 0.0f;

	//Global IDs
	uint32_t vectorID1 = tid;
	uint32_t matrixID1 = (sizeV * blockIdx.x) + tid;

	while (vectorID1 < sizeV) {
		sums[tid] += errors[vectorID1] * matrix[matrixID1];
		vectorID1 += blockDim.x;
		matrixID1 += blockDim.x;
	}

	__syncthreads();
	if (blockSize >= 1024u) { if (tid < 512u) sums[tid] += sums[tid + 512u]; __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) sums[tid] += sums[tid + 256u]; __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) sums[tid] += sums[tid + 128u]; __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) sums[tid] += sums[tid + 64u]; __syncthreads(); }

	if (tid < 32u) {
		volatile float* s = sums;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}
	if (tid == 0u) values[blockIdx.x] += AFD(values[blockIdx.x]) * (sums[0u] - errors[blockIdx.x]);
}

template<size_t blockSize>
__global__ void updateErrors(
	float* values,
	float* errors,
	float* matrix,
	const size_t sizeV
) {

	extern __shared__ float sums[];

	uint32_t tid = threadIdx.x;

	sums[tid] = 0.0f;

	//Global IDs
	uint32_t vectorID1 = tid;
	uint32_t matrixID1 = (sizeV * blockIdx.x) + tid;

	while (vectorID1 < sizeV) {
		sums[tid] += AF(values[vectorID1]) * matrix[matrixID1];
		vectorID1 += blockDim.x;
		matrixID1 += blockDim.x;
	}

	__syncthreads();
	if (blockSize >= 1024u) { if (tid < 512u) sums[tid] += sums[tid + 512u]; __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) sums[tid] += sums[tid + 256u]; __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) sums[tid] += sums[tid + 128u]; __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) sums[tid] += sums[tid + 64u]; __syncthreads(); }

	if (tid < 32u) {
		volatile float* s = sums;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}
	if (tid == 0u) errors[blockIdx.x] = AF(values[blockIdx.x]) - sums[0u];
}

__global__ void updateWeights(float* values, float* errors, float* weightsV, float* weightsE, float* valuesN, uint32_t sizeV) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t x = id % sizeV;
	uint32_t y = id / sizeV;
	float scale = valuesN[0u];
	weightsV[id] += scale * AF(values[y]) * errors[x];
	weightsE[id] += scale * AF(values[x]) * errors[y];
}

template<size_t activeSize>
NetworkCuda<activeSize>::NetworkCuda():
values(nullptr),
valuesN(nullptr),
errors(nullptr),
errorsN(nullptr),
weightsV(nullptr),
weightsE(nullptr)
{
	/*nThreads = s;
	nThreads = (nThreads > 1024u) ? 1024u : nThreads;
	nThreads--;
	nThreads |= nThreads >> 1u;
	nThreads |= nThreads >> 2u;
	nThreads |= nThreads >> 4u;
	nThreads |= nThreads >> 8u;
	nThreads |= nThreads >> 16u;
	nThreads |= nThreads >> 32u;
	nThreads++;
	
	nBlocks = (s / nThreads) + 1ull;*/


	cudaMalloc((void**)&values, activeSize * sizeof(float));
	cudaMalloc((void**)&valuesN, activeSize * sizeof(float));
	cudaMalloc((void**)&errors, activeSize * sizeof(float));
	cudaMalloc((void**)&errorsN, activeSize * sizeof(float));
	cudaMalloc((void**)&weightsV, getWeightSize() * sizeof(float));
	cudaMalloc((void**)&weightsE, getWeightSize() * sizeof(float));
	
};

template<size_t activeSize>
NetworkCuda<activeSize>::~NetworkCuda() {
	cudaFree(values);
	cudaFree(valuesN);
	cudaFree(errors);
	cudaFree(errorsN);
	cudaFree(weightsV);
	cudaFree(weightsE);
};

template<size_t activeSize>
float NetworkCuda<activeSize>::run() {
	updateErrors<1024u><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values, errors, weightsE, activeSize);
	updateValues<1024u><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values, errors, weightsV, activeSize);
	sumReduceInitial<1024u><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values, valuesN, activeSize);
#pragma unroll
	for (uint32_t i = activeSize / 1024u; i > 1024u; i >>= 1u) {
		sumReduceContinued<1024u><<<activeSize, 1024u, 1024u * sizeof(float)>>>(valuesN, valuesN, activeSize);
	}
	updateWeights<<<getGridSize(), 1024u >>>(values, errors, weightsV, weightsE, valuesN, activeSize);
	sumReduceInitial<1024u><<<activeSize, 1024u, 1024u * sizeof(float)>>>(errors, errorsN, activeSize);
#pragma unroll
	for (uint32_t i = activeSize / 1024u; i > 1024u; i >>= 1u) {
		sumReduceContinued<1024u><<<activeSize, 1024u, 1024u * sizeof(float)>>>(errorsN, errorsN, activeSize);
	}
	float error = 0.0f;
	cudaMemcpy(&error, errorsN, sizeof(float), cudaMemcpyDeviceToHost);
	return error;
};

//template<size_t activeSize>
//void NetworkCuda<activeSize>::testMatrixMultiplication() {
//	const static uint32_t sizesM = activeSize * activeSize;
//
//	float* testVector_d = nullptr;
//	float* testMatrix_d = nullptr;
//	float* testResults_d = nullptr;
//
//	float* testVector_h = nullptr;
//	float* testMatrix_h = nullptr;
//	float* testResults_h = nullptr;
//
//	float* testResults_dh = nullptr;
//
//	testVector_h = new float[activeSize];
//	testMatrix_h = new float[sizesM];
//	testResults_h = new float[activeSize];
//	testResults_dh = new float[activeSize];
//
//	std::srand(334);
//
//	for (uint32_t u = 0u; u < activeSize; u++) {
//		testVector_h[u] = ((float)std::rand() / (float)RAND_MAX);
//		testResults_h[u] = 0.0f;
//		for (uint32_t v = 0u; v < activeSize; v++) {
//			testMatrix_h[(activeSize * u) + v] = ((float)std::rand() / (float)RAND_MAX);
//		}
//	}
//	cudaMalloc((void**)&testVector_d, activeSize * sizeof(float));
//	cudaMalloc((void**)&testMatrix_d, sizesM * sizeof(float));
//	cudaMalloc((void**)&testResults_d, activeSize * sizeof(float));
//	cudaMemcpy(testVector_d, testVector_h, activeSize * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(testResults_d, testResults_h, activeSize * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(testMatrix_d, testMatrix_h, sizesM * sizeof(float), cudaMemcpyHostToDevice);
//	//cudaDeviceSynchronize();
//	for (uint32_t u = 0u; u < activeSize; u++) {
//		for (uint32_t v = 0u; v < activeSize; v++) {
//			testResults_h[u] += testVector_h[v] * testMatrix_h[(activeSize * u) + v];
//		}
//	}
//
//	delete[] testVector_h;
//	delete[] testMatrix_h;
//
//	updateVector<(activeSize > 1024u) ? 1024u : activeSize><<<activeSize, (activeSize > 1024u) ? 1024u : activeSize, ((activeSize > 1024u) ? 1024u : activeSize) * sizeof(float)>>>(testVector_d, testMatrix_d, testResults_d, activeSize, sizesM, activeSize);
//	
//	cudaMemcpy(testResults_dh, testResults_d, activeSize * sizeof(float), cudaMemcpyDeviceToHost);
//	//cudaDeviceSynchronize();
//	cudaFree(testVector_d);
//	cudaFree(testMatrix_d);
//	cudaFree(testResults_d);
//	float score = 0.0f;
//	for (uint32_t u = 0u; u < activeSize; u++) {
//		float sum = testResults_h[u] - testResults_dh[u];
//		score += sum * sum;
//	}
//
//	delete[] testResults_h;
//	delete[] testResults_dh;
//	
//	std::cout << "Matrix Kernel Test Results: " << score << '\n';
//};