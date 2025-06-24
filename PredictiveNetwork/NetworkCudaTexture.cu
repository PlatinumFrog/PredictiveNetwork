#include "NetworkCudaTexture.cuh"

#define GPU_ERROR_RET(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; return false; }
#define GPU_ERROR_ABT(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; abort(); }

constexpr float trainingRate = 1.8f;

__device__ float AF(float x) {
	return x / (1.0f + abs(x));
}

__device__ float AFD(float x) {
	float i = (1.0f + std::abs(x));
	return 1.0f / (i * i);
}

//// Sum reduction from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//// Then converted to vector magnitude calculation

template<size_t blockSize>
__global__ void sumReduceInitialAF(float* g_idata, float* g_odata, const size_t n) {
	extern __shared__ float sdata[];
	uint32_t tid = threadIdx.x;
	uint32_t i = blockIdx.x * blockSize + tid;
	uint32_t gridSize = blockSize * gridDim.x;
	sdata[tid] = 0.0f;
	while (i < n) {
		float x = AF(g_idata[i]);
		sdata[tid] += x * x;
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024u) { if (tid < 512u) { sdata[tid] += sdata[tid + 512u]; } __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) { sdata[tid] += sdata[tid + 256u]; } __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; } __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; } __syncthreads(); }
	if (tid < 32u) {
		volatile float* s = sdata;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}
	if (tid == 0u) g_odata[blockIdx.x] = sdata[0];
}

template<size_t blockSize>
__global__ void sumReduceInitial(float* g_idata, float* g_odata, const size_t n) {
	extern __shared__ float sdata[];
	uint32_t tid = threadIdx.x;
	uint32_t i = blockIdx.x * blockSize + tid;
	uint32_t gridSize = blockSize * gridDim.x;
	sdata[tid] = 0.0f;
	while (i < n) {
		float x = g_idata[i];
		sdata[tid] += x * x;
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024u) { if (tid < 512u) { sdata[tid] += sdata[tid + 512u]; } __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) { sdata[tid] += sdata[tid + 256u]; } __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; } __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; } __syncthreads(); }
	if (tid < 32u) {
		volatile float* s = sdata;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}
	if (tid == 0u) g_odata[blockIdx.x] = sdata[0];
}

template<size_t blockSize>
__global__ void sumReduceContinued(float* g_idata, float* g_odata, const size_t n) {
	extern __shared__ float sdata[];
	uint32_t tid = threadIdx.x;
	uint32_t i = blockIdx.x * blockSize + tid;
	uint32_t gridSize = blockSize * gridDim.x;
	sdata[tid] = 0.0f;
	while (i < n) {
		sdata[tid] += g_idata[i];
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024u) { if (tid < 512u) { sdata[tid] += sdata[tid + 512u]; } __syncthreads(); }
	if (blockSize >= 512u) { if (tid < 256u) { sdata[tid] += sdata[tid + 256u]; } __syncthreads(); }
	if (blockSize >= 256u) { if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; } __syncthreads(); }
	if (blockSize >= 128u) { if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; } __syncthreads(); }
	if (tid < 32u) {
		volatile float* s = sdata;
		if (blockSize >= 64u) s[tid] += s[tid + 32u];
		if (blockSize >= 32u) s[tid] += s[tid + 16u];
		if (blockSize >= 16u) s[tid] += s[tid + 8u];
		if (blockSize >= 8u) s[tid] += s[tid + 4u];
		if (blockSize >= 4u) s[tid] += s[tid + 2u];
		if (blockSize >= 2u) s[tid] += s[tid + 1u];
	}
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
	uint32_t vectorID = tid, matrixID = (sizeV * blockIdx.x) + tid;
	while (vectorID < sizeV) {
		sums[tid] += errors[vectorID] * matrix[matrixID];
		vectorID += blockSize, matrixID += blockSize;
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
	if (tid == 0u) {
		float v = values[blockIdx.x] - (trainingRate * AFD(values[blockIdx.x]) * sums[0u]);
		values[blockIdx.x] = (isnan(v) || isinf(v)) ? 0.0f : v;
	}
		
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
	uint32_t vectorID = tid, matrixID = (sizeV * blockIdx.x) + tid;
	sums[tid] = 0.0f;
	while (vectorID < sizeV) {
		sums[tid] += AF(values[vectorID]) * matrix[matrixID];
		vectorID += blockSize, matrixID += blockSize;
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
	if (tid == 0u) {
		float e = sums[0u];
		errors[blockIdx.x] = (isnan(e) || isinf(e)) ? 0.0f : e;
	}
}
__global__ void updateWeights(float* values, float* errors, float* weightsV, float* weightsE, float* valuesN, uint32_t sizeV, uint32_t sizeM) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < sizeM) {
		uint32_t x = id % sizeV;
		uint32_t y = id / sizeV;
		if (x == y) weightsE[id] = -1.0f, weightsV[id] = -1.0f;
		else {
			float scale = 1.0f / valuesN[0u];
			float w = weightsE[id] - trainingRate * scale * AF(values[x]) * errors[y];
			weightsE[id] = (isnan(w) || isinf(w)) ? 0.0f : w;
			w = weightsV[id] - trainingRate * scale * AF(values[y]) * errors[x];
			weightsV[id] = (isnan(w) || isinf(w)) ? 0.0f : w;
		}
	}
}

__global__ void setZero(float* values, size_t s) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < s) values[id] = 0.0f;
}

__global__ void setOnes(float* values, size_t s) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < s) values[id] = 1.0f;
}

__global__ void setValues(
	float* input,
	float* output,
	uint32_t* inputIDs,
	uint32_t s
) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < s) 
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
NetworkCudaTexture<activeSize>::NetworkCudaTexture():
	energy(0.0f),
	aabb{0.0f, 0.0f, 1.0f, 1.0f},
	valuesN(nullptr),
	errorsN(nullptr)
{
	cudaMalloc((void**)&valuesN, activeSize * sizeof(float));
	cudaMalloc((void**)&errorsN, activeSize * sizeof(float));
	const uint32_t wh = (uint32_t)std::ceil(std::sqrt(activeSize));
	values.setRect(-1.0, 0.0, 1.0, 1.0);
	values.setRes(wh, wh);
	errors.setRect(0.0, 0.0, 1.0, 1.0);
	errors.setRes(wh, wh);
	
	weightsV.setRect(-1.0, -1.0, 1.0, 1.0);
	weightsV.setRes(activeSize, activeSize);
	weightsE.setRect(0.0, -1.0, 1.0, 1.0);
	weightsE.setRes(activeSize, activeSize);

	reset();
};

template<size_t activeSize>
NetworkCudaTexture<activeSize>::~NetworkCudaTexture() {
	cudaFree(valuesN);
	cudaFree(errorsN);
};

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::reset() {
	resetValues();
	resetNValues();
	resetErrors();
	resetNErrors();
	resetMatrixValues();
	resetMatrixErrors();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::resetValues() {
	values.enableCuda();
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u>>>(values.data, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(values.data, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(values.data, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(values.data, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(values.data, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(values.data, activeSize);
	else setZero<<<1u, 32u>>>(values.data, activeSize);
	values.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::resetNValues() {
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (valuesN, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(valuesN, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(valuesN, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(valuesN, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(valuesN, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(valuesN, activeSize);
	else setZero<<<1u, 32u>>>(valuesN, activeSize);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::resetErrors() {
	errors.enableCuda();
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u>>>(errors.data, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(errors.data, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(errors.data, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(errors.data, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(errors.data, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(errors.data, activeSize);
	else setZero<<<1u, 32u>>>(errors.data, activeSize);
	errors.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::resetNErrors() {
	if (activeSize > 1024u) setZero << <(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (errorsN, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(errorsN, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(errorsN, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(errorsN, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(errorsN, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(errorsN, activeSize);
	else setZero<<<1u, 32u>>>(errorsN, activeSize);
}      

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::resetMatrixValues() {
	weightsV.enableCuda();
	/*float* r = new float[activeSize * activeSize];
	for (uint32_t u = 0u; u < activeSize * activeSize; u++) r[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	cudaMemcpy(weightsV.data, r, activeSize * activeSize * sizeof(float), cudaMemcpyHostToDevice);*/
	setZero<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsV.data, activeSize * activeSize);
	//delete[] r;
	weightsV.disableCuda();
}
template<size_t activeSize>
void NetworkCudaTexture<activeSize>::resetMatrixErrors() {
	weightsE.enableCuda();
	/*float* r = new float[activeSize * activeSize];
	for (uint32_t u = 0u; u < activeSize * activeSize; u++) r[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	cudaMemcpy(weightsE.data, r, activeSize * activeSize * sizeof(float), cudaMemcpyHostToDevice);*/
	setZero<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsE.data, activeSize * activeSize);
	//delete[] r;
	weightsE.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::train(
	float* input,
	float* output,
	uint32_t* inputIDs,
	uint32_t* outputIDs,
	uint32_t sizeI,
	uint32_t sizeO
) {
	float* ptrV = nullptr;
	uint32_t* ptrI = nullptr;

	values.enableCuda();
	errors.enableCuda();
	weightsV.enableCuda();
	weightsE.enableCuda();

	cudaMalloc(&ptrV, sizeI * sizeof(float));
	cudaMalloc(&ptrI, sizeI * sizeof(uint32_t));
	cudaMemcpy(ptrV, input, sizeI * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ptrI, inputIDs, sizeI * sizeof(uint32_t), cudaMemcpyHostToDevice);
	setValues<<<((sizeI - 1u) >> 5u) + 1u, 32u>>>(ptrV, values.data, ptrI, sizeI);
	cudaFree(ptrV);
	cudaFree(ptrI);
	cudaMalloc(&ptrV, sizeO * sizeof(float));
	cudaMalloc(&ptrI, sizeO * sizeof(uint32_t));
	cudaMemcpy(ptrV, output, sizeO * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ptrI, outputIDs, sizeO * sizeof(uint32_t), cudaMemcpyHostToDevice);
	setValues<<<((sizeO - 1u) >> 5u) + 1u, 32u>>> (ptrV, values.data, ptrI, sizeO);
	cudaFree(ptrV);
	cudaFree(ptrI);
	
	calcErrors();
	
	resetNValues();
	calcNormAF(values.data, valuesN);

	calcValues();
	
	

	calcWeights();

	resetNErrors();
	calcNorm(errors.data, errorsN);

	values.disableCuda();
	errors.disableCuda();
	weightsV.disableCuda();
	weightsE.disableCuda();

	cudaMemcpy(&energy, errorsN, sizeof(float), cudaMemcpyDeviceToHost);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::run(
	float* input,
	uint32_t* inputIDs,
	uint32_t sizeI
) {
	float* ptrV = nullptr;
	uint32_t* ptrI = nullptr;

	values.enableCuda();
	errors.enableCuda();
	weightsV.enableCuda();
	weightsE.enableCuda();

	cudaMalloc(&ptrV, sizeI * sizeof(float));
	cudaMalloc(&ptrI, sizeI * sizeof(uint32_t));
	cudaMemcpy(ptrV, input, sizeI * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ptrI, inputIDs, sizeI * sizeof(uint32_t), cudaMemcpyHostToDevice);
	setValues<<<((sizeI - 1u) >> 5u) + 1u, 32u>>>(ptrV, values.data, ptrI, sizeI);
	cudaFree(ptrV);
	cudaFree(ptrI);

	calcErrors();
	calcValues();

	resetNErrors();
	calcNorm(errors.data, errorsN);

	values.disableCuda();
	errors.disableCuda();
	weightsV.disableCuda();
	weightsE.disableCuda();

	cudaMemcpy(&energy, errorsN, sizeof(float), cudaMemcpyDeviceToHost);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::sleep() {

	values.enableCuda();
	errors.enableCuda();
	weightsV.enableCuda();
	weightsE.enableCuda();

	calcErrors();
	calcValues();
	resetNValues();
	calcNormAF(values.data, valuesN);
	calcWeights();
	resetNErrors();
	calcNorm(errors.data, errorsN);

	values.disableCuda();
	errors.disableCuda();
	weightsV.disableCuda();
	weightsE.disableCuda();

	cudaMemcpy(&energy, errorsN, sizeof(float), cudaMemcpyDeviceToHost);
}


template<size_t activeSize>
inline void NetworkCudaTexture<activeSize>::calcNorm(float* input, float* output) {
	if(activeSize > 1024u) sumReduceInitial<1024u><<<(((activeSize - 1u) >> 10u) + 1u), 1024u, 1024u * sizeof(float)>>> (input, output, activeSize);
	else if(activeSize > 512u) sumReduceInitial<512u><<<1u, 512u, 512u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 256u) sumReduceInitial<256u><<<1u, 256u, 256u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 128u) sumReduceInitial<128u><<<1u, 128u, 128u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 64u) sumReduceInitial<64u><<<1u, 64u, 64u * sizeof(float)>>>(input, output, activeSize);
	else sumReduceInitial<32u><<<1u, 32u, 32u * sizeof(float)>>>(input, output, activeSize);
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
inline void NetworkCudaTexture<activeSize>::calcNormAF(float* input, float* output) {
	if (activeSize > 1024u) sumReduceInitialAF<1024ull><<<(((activeSize - 1u) >> 10u) + 1u), 1024u, 1024u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 512u) sumReduceInitialAF<512ull><<<1u, 512u, 512u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 256u) sumReduceInitialAF<256ull><<<1u, 256u, 256u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 128u) sumReduceInitialAF<128ull><<<1u, 128u, 128u * sizeof(float)>>>(input, output, activeSize);
	else if (activeSize > 64u) sumReduceInitialAF<64ull><<<1u, 64u, 64u * sizeof(float)>>>(input, output, activeSize);
	else sumReduceInitialAF<32ull><<<1u, 32u, 32u * sizeof(float)>>>(input, output, activeSize);
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
inline void NetworkCudaTexture<activeSize>::calcValues() {
	if (activeSize > 1024u) updateValues<1024ull><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values.data, errors.data, weightsV.data, activeSize);
	else if (activeSize > 512u) updateValues<512ull><<<activeSize, 512u, 512u * sizeof(float)>>>(values.data, errors.data, weightsV.data, activeSize);
	else if (activeSize > 256u) updateValues<256ull><<<activeSize, 256u, 256u * sizeof(float)>>>(values.data, errors.data, weightsV.data, activeSize);
	else if (activeSize > 128u) updateValues<128ull><<<activeSize, 128u, 128u * sizeof(float)>>>(values.data, errors.data, weightsV.data, activeSize);
	else if (activeSize > 64u) updateValues<64ull><<<activeSize, 64u, 64u * sizeof(float)>>>(values.data, errors.data, weightsV.data, activeSize);
	else updateValues<32ull><<<activeSize, 32u, 32u * sizeof(float)>>>(values.data, errors.data, weightsV.data, activeSize);
}

template<size_t activeSize>
inline void NetworkCudaTexture<activeSize>::calcErrors() {
	if (activeSize > 1024u) updateErrors<1024ull><<<activeSize, 1024u, 1024u * sizeof(float)>>>(values.data, errors.data, weightsE.data, activeSize);
	else if (activeSize > 512u) updateErrors<512ull><<<activeSize, 512u, 512u * sizeof(float)>>>(values.data, errors.data, weightsE.data, activeSize);
	else if (activeSize > 256u) updateErrors<256ull><<<activeSize, 256u, 256u * sizeof(float)>>>(values.data, errors.data, weightsE.data, activeSize);
	else if (activeSize > 128u) updateErrors<128ull><<<activeSize, 128u, 128u * sizeof(float)>>>(values.data, errors.data, weightsE.data, activeSize);
	else if (activeSize > 64u) updateErrors<64ull><<<activeSize, 64u, 64u * sizeof(float)>>>(values.data, errors.data, weightsE.data, activeSize);
	else updateErrors<32ull><<<activeSize, 32u, 32u * sizeof(float)>>>(values.data, errors.data, weightsE.data, activeSize);
}
template<size_t activeSize>
inline void NetworkCudaTexture<activeSize>::calcWeights() {
	if (activeSize * activeSize > 1024u) updateWeights<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(values.data, errors.data, weightsV.data, weightsE.data, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 512u) updateWeights<<<1u, 512u>>>(values.data, errors.data, weightsV.data, weightsE.data, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 256u) updateWeights<<<1u, 256u>>>(values.data, errors.data, weightsV.data, weightsE.data, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 128u) updateWeights<<<1u, 128u>>>(values.data, errors.data, weightsV.data, weightsE.data, valuesN, activeSize, activeSize * activeSize);
	else if (activeSize * activeSize > 64u) updateWeights<<<1u, 64u>>>(values.data, errors.data, weightsV.data, weightsE.data, valuesN, activeSize, activeSize * activeSize);
	else updateWeights<<<1u, 32u>>>(values.data, errors.data, weightsV.data, weightsE.data, valuesN, activeSize, activeSize * activeSize);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::CudaEnableValues() {
	values.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::CudaEnableErrors() {
	errors.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::CudaEnableWeights() {
	weightsV.enableCuda();
	weightsE.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::CudaDisableValues() {
	values.disableCuda();
}
template<size_t activeSize>
void NetworkCudaTexture<activeSize>::CudaDisableErrors() {
	errors.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::CudaDisableWeights() {
	weightsV.disableCuda();
	weightsE.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::draw() {
	weightsE.updateTexture();
	weightsE.draw();
	weightsV.updateTexture();
	weightsV.draw();
	errors.updateTexture();
	errors.draw();
	values.updateTexture();
	values.draw();
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