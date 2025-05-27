#include "NetworkCuda.cuh"

#define GPU_ERROR_RET(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; return false; }
#define GPU_ERROR_ABT(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; abort(); }

__device__ float AF(float x) {
	return x / (1.0f + std::abs(x));
}

__device__ float AFD(float x) {
	float i = (1.0f + std::abs(x));
	return 1.0f / (i * i);
}

//// Sum reduction from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//// Then converted to vector magnitude calculation

template<size_t blockSize>
__global__ void sumReduceInitialAF(float* g_idata, float* g_odata, const size_t n) {
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
__global__ void sumReduceInitial(float* g_idata, float* g_odata, const size_t n) {
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
		float x = g_idata[i];
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
__global__ void sumReduceContinued(float* g_idata, float* g_odata, const size_t n) {
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

__global__ void updateWeights(float* values, float* errors, float* weightsV, float* weightsE, float* valuesN, uint32_t sizeV, uint32_t sizeM) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < sizeM) {
		uint32_t x = id % sizeV;
		uint32_t y = id / sizeV;
		float scale = 1.0f / valuesN[0u];
		weightsV[id] += scale * AF(values[y]) * errors[x];
		weightsE[id] += scale * AF(values[x]) * errors[y];
	}
}

__global__ void setZero(float* values, size_t s) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < s) values[id] = 0.0f;
}

__global__ void setValues(
	float* input,
	float* output,
	uint32_t* inputIDs,
	uint32_t s
) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	output[inputIDs[id]] = input[id];
}

__global__ void setNodePos(float* positions, size_t size) {
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	float j = (float)(i) * (TAU / (float)size);
	float x = std::cos(j);
	float y = std::sin(j);
	uint32_t id = 2u * i;
	positions[id] = x;
	positions[id + 1u] = y;
}

__global__ void setWeightPos(float* posW, float* posN, uint32_t sizeV, uint32_t sizeM) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < sizeM) {
		uint32_t x1 = 2u * (id % sizeV);
		uint32_t x2 = 2u * (id / sizeV);
		uint32_t y1 = x1 + 1u;
		uint32_t y2 = x2 + 1u;
		uint32_t idw = 4u * id;
		posW[idw] = posN[x1];
		posW[idw + 1u] = posN[y1];
		posW[idw + 2u] = posN[x2];
		posW[idw + 3u] = posN[y2];
	}
}

template<size_t activeSize>
NetworkCuda<activeSize>::NetworkCuda():
	values(nullptr),
	valuesN(nullptr),
	errors(nullptr),
	errorsN(nullptr),
	weightsV(nullptr),
	weightsE(nullptr),
	nodePos(nullptr)
{
	//cudaMalloc((void**)&values, activeSize * sizeof(float));
	cudaMalloc((void**)&valuesN, activeSize * sizeof(float));
	//cudaMalloc((void**)&errors, activeSize * sizeof(float));
	cudaMalloc((void**)&errorsN, activeSize * sizeof(float));
	//cudaMalloc((void**)&weightsV, activeSize * activeSize * sizeof(float));
	//cudaMalloc((void**)&weightsE, activeSize * activeSize * sizeof(float));
	//cudaMalloc((void**)&nodePos, 2u * activeSize * sizeof(float));
	
	glGenBuffers(1u, &nodePosVBO);
	glGenBuffers(1u, &nodeValVBO);
	glGenBuffers(1u, &nodeErrVBO);
	glGenVertexArrays(1u, &nodeVAO);
	glBindVertexArray(nodeVAO);
	glBindBuffer(GL_ARRAY_BUFFER, nodePosVBO);
	glBufferData(GL_ARRAY_BUFFER, 2u * activeSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, nodeValVBO);
	glBufferData(GL_ARRAY_BUFFER, activeSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, nodeErrVBO);
	glBufferData(GL_ARRAY_BUFFER, activeSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0u);
	glBindVertexArray(0u);

	glGenBuffers(1u, &weightPosVBO);
	glGenBuffers(1u, &weightValVBO);
	glGenBuffers(1u, &weightErrVBO);
	glGenVertexArrays(1u, &weightVAO);
	glBindVertexArray(weightVAO);
	glBindBuffer(GL_ARRAY_BUFFER, weightPosVBO);
	glBufferData(GL_ARRAY_BUFFER, 4u * activeSize * activeSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, weightValVBO);
	glBufferData(GL_ARRAY_BUFFER, activeSize * activeSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, weightErrVBO);
	glBufferData(GL_ARRAY_BUFFER, activeSize * activeSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0u);
	glBindVertexArray(0u);

	cudaGraphicsGLRegisterBuffer(&cgrNPositn, nodePosVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cgrNValues, nodeValVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cgrNErrors, nodeErrVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cgrWPositn, weightPosVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cgrWValues, weightValVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cgrWErrors, weightErrVBO, cudaGraphicsRegisterFlagsWriteDiscard);


	nodeShader = Shader::create("nodeShader.vert", "nodeShader.geom", "nodeShader.frag");
	weightShader = Shader::create("weightShader.vert", "weightShader.geom", "weightShader.frag");
	
	CudaEnableValues();
	CudaEnableErrors();
	CudaEnableWeights();
	reset();
	CudaDisableErrors();
	CudaDisableValues();
	CudaDisableWeights();

	CudaEnableNodePos();
	resetNodePositions();
	CudaDisableNodePos();

	CudaEnableWeightPos();
	resetWeightPositions();
	CudaDisableWeightPos();
};

template<size_t activeSize>
NetworkCuda<activeSize>::~NetworkCuda() {
	cudaGraphicsUnregisterResource(cgrNPositn);
	cudaGraphicsUnregisterResource(cgrNValues);
	cudaGraphicsUnregisterResource(cgrNErrors);
	glDeleteBuffers(1, &nodePosVBO);
	glDeleteBuffers(1, &nodeValVBO);
	glDeleteBuffers(1, &nodeErrVBO);
	glDeleteVertexArrays(1, &nodeVAO);

	cudaGraphicsUnregisterResource(cgrWPositn);
	cudaGraphicsUnregisterResource(cgrWValues);
	cudaGraphicsUnregisterResource(cgrWErrors);
	glDeleteBuffers(1, &weightPosVBO);
	glDeleteBuffers(1, &weightValVBO);
	glDeleteBuffers(1, &weightErrVBO);
	glDeleteVertexArrays(1, &weightVAO);

	//cudaFree(values);
	cudaFree(valuesN);
	//cudaFree(errors);
	cudaFree(errorsN);
	//cudaFree(weightsV);
	//cudaFree(weightsE);
	//cudaFree(nodePos);
};

template<size_t activeSize>
void NetworkCuda<activeSize>::reset() {
	resetValues();
	resetNValues();
	resetErrors();
	resetNErrors();
	resetMatrixValues();
	resetMatrixErrors();
}
template<size_t activeSize>
void NetworkCuda<activeSize>::resetNodePositions() {
	if (activeSize > 1024u) setNodePos<<<(((activeSize - 1u) >> 10u) + 1u), 1024u>>>(nodePos, activeSize);
	else if (activeSize > 512u) setNodePos<<<1u, 1024u>>>(nodePos, activeSize);
	else if (activeSize > 256u) setNodePos<<<1u, 512u>>>(nodePos, activeSize);
	else if (activeSize > 128u) setNodePos<<<1u, 256u>>>(nodePos, activeSize);
	else if (activeSize > 64u) setNodePos<<<1u, 128u>>>(nodePos, activeSize);
	else if (activeSize > 32u) setNodePos<<<1u, 64u>>>(nodePos, activeSize);
	else setNodePos<<<1u, 32u>>>(nodePos, activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::resetWeightPositions() {
	setWeightPos<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightPos, nodePos, activeSize, activeSize * activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::resetValues() {
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u>>>(values, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(values, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(values, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(values, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(values, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(values, activeSize);
	else setZero<<<1u, 32u>>>(values, activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::resetNValues() {
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (valuesN, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(valuesN, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(valuesN, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(valuesN, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(valuesN, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(valuesN, activeSize);
	else setZero<<<1u, 32u>>>(valuesN, activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::resetErrors() {
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u>>>(errors, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(errors, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(errors, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(errors, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(errors, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(errors, activeSize);
	else setZero<<<1u, 32u>>>(errors, activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::resetNErrors() {
	if (activeSize > 1024u) setZero << <(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (errorsN, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(errorsN, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(errorsN, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(errorsN, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(errorsN, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(errorsN, activeSize);
	else setZero<<<1u, 32u>>>(errorsN, activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::resetMatrixValues() {
	setZero<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsV, activeSize * activeSize);
}
template<size_t activeSize>
void NetworkCuda<activeSize>::resetMatrixErrors() {
	setZero<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsE, activeSize * activeSize);
}

template<size_t activeSize>
float NetworkCuda<activeSize>::train(
	float* input,
	float* output,
	uint32_t* inputIDs,
	uint32_t* outputIDs,
	uint32_t sizeI,
	uint32_t sizeO
) {
	float* ptrV = nullptr;
	uint32_t* ptrI = nullptr;

	CudaEnableValues();
	CudaEnableErrors();
	CudaEnableWeights();

	cudaMalloc(&ptrV, sizeI * sizeof(float));
	cudaMalloc(&ptrI, sizeI * sizeof(uint32_t));
	cudaMemcpy(ptrV, input, sizeI * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ptrI, inputIDs, sizeI * sizeof(uint32_t), cudaMemcpyHostToDevice);
	setValues<<<((sizeI - 1u) >> 5u) + 1u, 32u>>>(ptrV, values, ptrI, sizeI);
	cudaMemcpy(ptrV, output, sizeO * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ptrI, outputIDs, sizeO * sizeof(uint32_t), cudaMemcpyHostToDevice);
	setValues<<<((sizeO - 1u) >> 5u) + 1u, 32u>>> (ptrV, values, ptrI, sizeO);
	cudaFree(ptrV);
	cudaFree(ptrI);

	resetNValues();
	calcNormAF(values, valuesN);
	calcErrors();
	calcValues();
	calcWeights();
	resetNErrors();
	calcNorm(errors, errorsN);

	CudaDisableErrors();
	CudaDisableValues();
	CudaDisableWeights();

	float error = 0.0f;
	cudaMemcpy(&error, errorsN, sizeof(float), cudaMemcpyDeviceToHost);
	return error;
}


template<size_t activeSize>
inline void NetworkCuda<activeSize>::calcNorm(float* input, float* output) {
	if(activeSize > 1024u) sumReduceInitial<1024u><<<(((activeSize - 1u) >> 10u) + 1u), 1024u, 1024u * sizeof(float)>>> (input, output, activeSize);
	else if(activeSize > 512u) sumReduceInitial<512u><<<1u, 512u, 512u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 256u) sumReduceInitial<256u><<<1u, 256u, 256u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 128u) sumReduceInitial<128u><<<1u, 128u, 128u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 64u) sumReduceInitial<64u><<<1u, 64u, 64u * sizeof(float)>>>(input, output, activeSize);
	else sumReduceInitial<32u><<<1u, 32u, 32u * sizeof(float)>>>(input, output, activeSize);
#pragma unroll
	for (size_t i = (((activeSize - 1u) >> 10u) + 1u); i > 1u; i = (((i - 1u) >> 10u) + 1u)) {
		if (i > 1024u) sumReduceContinued<1024u><<<(((i - 1u) >> 10u) + 1u), 1024u, 1024u * sizeof(float)>>>(output, output, activeSize);
		else if (i > 512u) sumReduceContinued<512u><<<1u, 512u, 512u * sizeof(float)>>>(output, output, i);
		else if (i > 256u) sumReduceContinued<256u><<<1u, 256u, 256u * sizeof(float)>>>(output, output, i);
		else if (i > 128u) sumReduceContinued<128u><<<1u, 128u, 128u * sizeof(float)>>>(output, output, i);
		else if (i > 64u) sumReduceContinued<64u><<<1u, 64u, 64u * sizeof(float)>>> (output, output, i);
		else sumReduceContinued<32u><<<1u, 32u, 32u * sizeof(float)>>>(output, output, i);
	}
}

template<size_t activeSize>
inline void NetworkCuda<activeSize>::calcNormAF(float* input, float* output) {
	if (activeSize > 1024u) sumReduceInitialAF<1024ull><<<(((activeSize - 1u) >> 10u) + 1u), 1024u, 1024u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 512u) sumReduceInitialAF<512ull><<<1u, 512u, 512u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 256u) sumReduceInitialAF<256ull><<<1u, 256u, 256u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 128u) sumReduceInitialAF<128ull><<<1u, 128u, 128u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 64u) sumReduceInitialAF<64ull><<<1u, 64u, 64u * sizeof(float)>>>(input, output, activeSize);
	else sumReduceInitialAF<32ull><<<1u, 32u, 32u * sizeof(float)>>>(input, output, activeSize);
#pragma unroll
	for (size_t i = (((activeSize - 1u) >> 10u) + 1u); i > 1u; i = (((i - 1u) >> 10u) + 1u)) {
		if (i > 1024u) sumReduceContinued<1024ull><<<(((i - 1u) >> 10u) + 1u), 1024u, 1024u * sizeof(float)>>>(output, output, i);
		else if (i > 512u) sumReduceContinued<512ull><<<1u, 512u, 512u * sizeof(float)>>>(output, output, i);
		else if (i > 256u) sumReduceContinued<256ull><<<1u, 256u, 256u * sizeof(float)>>>(output, output, i);
		else if (i > 128u) sumReduceContinued<128ull><<<1u, 128u, 128u * sizeof(float)>>>(output, output, i);
		else if (i > 64u) sumReduceContinued<64ull><<<1u, 64u, 64u * sizeof(float)>>>(output, output, i);
		else sumReduceContinued<32ull><<<1u, 32u, 32u * sizeof(float)>>>(output, output, i);
	}
}
template<size_t activeSize>
inline void NetworkCuda<activeSize>::calcValues() {
	if (activeSize > 1024u) updateValues<1024ull><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values, errors, weightsV, activeSize);
	else if (activeSize > 512u) updateValues<512ull><<<activeSize, 512u, 512u * sizeof(float)>>>(values, errors, weightsV, activeSize);
	else if (activeSize > 256u) updateValues<256ull><<<activeSize, 256u, 256u * sizeof(float)>>>(values, errors, weightsV, activeSize);
	else if (activeSize > 128u) updateValues<128ull><<<activeSize, 128u, 128u * sizeof(float)>>>(values, errors, weightsV, activeSize);
	else if (activeSize > 64u) updateValues<64ull><<<activeSize, 64u, 64u * sizeof(float)>>>(values, errors, weightsV, activeSize);
	else updateValues<32ull><<<activeSize, 32u, 32u * sizeof(float)>>>(values, errors, weightsV, activeSize);
}

template<size_t activeSize>
inline void NetworkCuda<activeSize>::calcErrors() {
	if (activeSize > 1024u) updateErrors<1024ull><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values, errors, weightsE, activeSize);
	else if (activeSize > 512u) updateErrors<512ull><<<activeSize, 512u, 512u * sizeof(float)>>>(values, errors, weightsE, activeSize);
	else if (activeSize > 256u) updateErrors<256ull><<<activeSize, 256u, 256u * sizeof(float)>>>(values, errors, weightsE, activeSize);
	else if (activeSize > 128u) updateErrors<128ull><<<activeSize, 128u, 128u * sizeof(float)>>>(values, errors, weightsE, activeSize);
	else if (activeSize > 64u) updateErrors<64ull><<<activeSize, 64u, 64u * sizeof(float)>>>(values, errors, weightsE, activeSize);
	else updateErrors<32ull><<<activeSize, 32u, 32u * sizeof(float)>>>(values, errors, weightsE, activeSize);
}
template<size_t activeSize>
inline void NetworkCuda<activeSize>::calcWeights() {
	if (activeSize * activeSize > 1024u) updateWeights<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(values, errors, weightsV, weightsE, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 512u) updateWeights<<<1u, 512u>>>(values, errors, weightsV, weightsE, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 256u) updateWeights<<<1u, 256u>>>(values, errors, weightsV, weightsE, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 128u) updateWeights<<<1u, 128u>>>(values, errors, weightsV, weightsE, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 64u) updateWeights<<<1u, 64u>>>(values, errors, weightsV, weightsE, valuesN, activeSize, activeSize * activeSize);
	else updateWeights<<<1u, 32u>>>(values, errors, weightsV, weightsE, valuesN, activeSize, activeSize * activeSize);
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaEnableValues() {
	if (!(flags & 1u)) {
		size_t size;
		cudaGraphicsMapResources(1, &cgrNValues, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&values, &size, cgrNValues);
		flags |= 1u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaEnableErrors() {
	if (!(flags & 2u)) {
		size_t size;
		cudaGraphicsMapResources(1, &cgrNErrors, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&errors, &size, cgrNErrors);
		flags |= 2u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaEnableNodePos() {
	if (!(flags & 4u)) {
		size_t size;
		cudaGraphicsMapResources(1, &cgrNPositn, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&nodePos, &size, cgrNPositn);
		flags |= 4u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaEnableWeights() {
	if (!(flags & 8u)) {
		size_t size;
		cudaGraphicsMapResources(1, &cgrWValues, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&weightsV, &size, cgrWValues);
		cudaGraphicsMapResources(1, &cgrWErrors, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&weightsE, &size, cgrWErrors);
		flags |= 8u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaEnableWeightPos() {
	if (!(flags & 16u)) {
		size_t size;
		cudaGraphicsMapResources(1, &cgrWPositn, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&weightPos, &size, cgrWPositn);
		flags |= 16u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaDisableValues() {
	if (flags & 1u) {
		cudaGraphicsUnmapResources(1, &cgrNValues, 0);
		flags &= ~1u;
	}
}
template<size_t activeSize>
void NetworkCuda<activeSize>::CudaDisableErrors() {
	if (flags & 2u) {
		cudaGraphicsUnmapResources(1, &cgrNErrors, 0);
		flags &= ~2u;
	}
}
template<size_t activeSize>
void NetworkCuda<activeSize>::CudaDisableNodePos() {
	if (flags & 4u) {
		cudaGraphicsUnmapResources(1, &cgrNPositn, 0);
		flags &= ~4u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaDisableWeights() {
	if (flags & 8u) {
		size_t size;
		cudaGraphicsUnmapResources(1, &cgrWValues, 0);
		cudaGraphicsUnmapResources(1, &cgrWErrors, 0);
		flags &= ~8u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::CudaDisableWeightPos() {
	if (flags & 16u) {
		cudaGraphicsUnmapResources(1, &cgrWPositn, 0);
		flags &= ~16u;
	}
}

template<size_t activeSize>
void NetworkCuda<activeSize>::draw() {
	glUseProgram(nodeShader);
	glBindVertexArray(nodeVAO);
	glUniform4f(3, 0.0f, 0.0f, minDrawRadius(), 1.0f);
	glDrawArrays(GL_POINTS, 0, activeSize);

	glUseProgram(weightShader);
	glBindVertexArray(weightVAO);
	glUniform4f(3, 0.0f, 0.0f, minDrawRadius(), 1.0f);
	glDrawArrays(GL_POINTS, 0, activeSize * activeSize);

}

//template<size_t activeSize>
//void NetworkCuda<activeSize>::testVectorNormalization() {
//	std::srand(3728172);
//	float testVector_h[activeSize];
//	for (size_t i = 0ull; i < activeSize; i++) testVector_h[i] = 2.0f * ((float)std::rand() / (float)RAND_MAX) - 1.0f;
//	float result_h = 0.0f;
//	for (size_t i = 0ull; i < activeSize; i++) {
//		result_h += testVector_h[i] * testVector_h[i];
//	}
//	float* testVector_d = nullptr;
//	float* testVectorN_d = nullptr;
//	cudaMalloc((void**)&testVector_d, activeSize * sizeof(float));
//	cudaMalloc((void**)&testVectorN_d, activeSize * sizeof(float));
//	cudaMemcpy(testVector_d, testVector_h, activeSize * sizeof(float), cudaMemcpyHostToDevice);
//	setZero<<<((activeSize + 1u) >> 10u), 1024u>>>(testVectorN_d, activeSize);
//	calcNorm(testVector_d, testVectorN_d);
//	float result_d = 0.0f;
//	cudaMemcpy(&result_d, testVectorN_d, sizeof(float), cudaMemcpyDeviceToHost);
//	std::cout << "Host Result: " << result_h << ", Device Result: " << result_d << '\n';
//	for (size_t i = 0ull; i < activeSize; i++) testVector_h[i] = 2.0f * ((float)std::rand() / (float)RAND_MAX) - 1.0f;
//	result_h = 0.0f;
//	for (size_t i = 0ull; i < activeSize; i++) {
//		result_h += testVector_h[i] * testVector_h[i];
//	}
//	cudaMemcpy(testVector_d, testVector_h, activeSize * sizeof(float), cudaMemcpyHostToDevice);
//	setZero << <((activeSize + 1u) >> 10u), 1024u >> > (testVectorN_d, activeSize);
//	calcNorm(testVector_d, testVectorN_d);
//	result_d = 0.0f;
//	cudaMemcpy(&result_d, testVectorN_d, sizeof(float), cudaMemcpyDeviceToHost);
//	std::cout << "Host Result: " << result_h << ", Device Result: " << result_d << '\n';
//	cudaFree(testVector_d);
//	cudaFree(testVectorN_d);
//};

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