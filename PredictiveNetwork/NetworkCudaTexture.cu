#include "NetworkCudaTexture.cuh"

#define GPU_ERROR_RET(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; return false; }
#define GPU_ERROR_ABT(msg, err) if(err != cudaSuccess) { std::cerr << "\n>> " __FILE__ " at line " << __LINE__ << ":\n<< " #msg << ": " << cudaGetErrorString(err) << std::endl; abort(); }

constexpr float trainingRate = 0.25f;

__device__ float AF(float x) {
	return 1.0f / (1.0f + (x * x));
}

__device__ float AFD(float x) {
	float i = 1.0f + (x * x);
	return -2.0f * (x / (i * i));
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
	if(id < s) output[inputIDs[id]] = input[id];
}


__global__ void displaceData(float* data, float* disp, size_t size) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < size) data[id] += disp[id];
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

	zero();
};

template<size_t activeSize>
NetworkCudaTexture<activeSize>::NetworkCudaTexture(NetworkCudaTexture& n):
	energy(n.energy),
	aabb(n.aabb),
	valuesN(nullptr),
	errorsN(nullptr)
{
	cudaMalloc((void**)&valuesN, activeSize * sizeof(float));
	cudaMalloc((void**)&errorsN, activeSize * sizeof(float));
	cudaMemcpy(valuesN, n.valuesN, activeSize * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(errorsN, n.errorsN, activeSize * sizeof(float), cudaMemcpyDeviceToDevice);
	
	values = n.values;
	errors = n.errors;
	weightsV = n.weightsV;
	weightsE = n.weightsE;

	

};

template<size_t activeSize>
NetworkCudaTexture<activeSize>::NetworkCudaTexture(float4 ab):
	energy(0.0f),
	aabb(ab),
	valuesN(nullptr),
	errorsN(nullptr) 
{
	cudaMalloc((void**)&valuesN, activeSize * sizeof(float));
	cudaMalloc((void**)&errorsN, activeSize * sizeof(float));
	const uint32_t wh = (uint32_t)std::ceil(std::sqrt(activeSize));
	values.setRes(wh, wh);
	errors.setRes(wh, wh);
	weightsV.setRes(activeSize, activeSize);
	weightsE.setRes(activeSize, activeSize);
	setAABB(n.aabb);
};

template<size_t activeSize>
NetworkCudaTexture<activeSize>::~NetworkCudaTexture() {
	cudaFree(valuesN);
	cudaFree(errorsN);
};

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::zero() {
	zeroValues();
	zeroNValues();
	zeroErrors();
	zeroNErrors();
	zeroMatrixValues();
	zeroMatrixErrors();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::zeroValues() {
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
void NetworkCudaTexture<activeSize>::zeroNValues() {
	if (activeSize > 1024u) setZero<<<(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (valuesN, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(valuesN, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(valuesN, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(valuesN, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(valuesN, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(valuesN, activeSize);
	else setZero<<<1u, 32u>>>(valuesN, activeSize);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::zeroErrors() {
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
void NetworkCudaTexture<activeSize>::zeroNErrors() {
	if (activeSize > 1024u) setZero << <(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (errorsN, activeSize);
	else if (activeSize > 512u) setZero<<<1u, 1024u>>>(errorsN, activeSize);
	else if (activeSize > 256u) setZero<<<1u, 512u>>>(errorsN, activeSize);
	else if (activeSize > 128u) setZero<<<1u, 256u>>>(errorsN, activeSize);
	else if (activeSize > 64u) setZero<<<1u, 128u>>>(errorsN, activeSize);
	else if (activeSize > 32u) setZero<<<1u, 64u>>>(errorsN, activeSize);
	else setZero<<<1u, 32u>>>(errorsN, activeSize);
}      

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::zeroMatrixValues() {
	weightsV.enableCuda();
	setZero<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsV.data, activeSize * activeSize);
	weightsV.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::zeroMatrixErrors() {
	weightsE.enableCuda();
	setZero<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsE.data, activeSize * activeSize);
	weightsE.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::randomize() {
	randomizeValues();
	randomizeErrors();
	randomizeMatrixValues();
	randomizeMatrixErrors();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::randomizeValues() {
	values.enableCuda();
	float* r = new float[activeSize];
	for (uint32_t u = 0u; u < activeSize; u++) r[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	cudaMemcpy(values.data, r, activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] r;
	values.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::randomizeErrors() {
	errors.enableCuda();
	float* r = new float[activeSize];
	for (uint32_t u = 0u; u < activeSize; u++) r[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	cudaMemcpy(errors.data, r, activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] r;
	errors.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::randomizeMatrixValues() {
	weightsV.enableCuda();
	float* r = new float[activeSize * activeSize];
	for (uint32_t u = 0u; u < activeSize * activeSize; u++) r[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	cudaMemcpy(weightsV.data, r, activeSize * activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] r;
	weightsV.disableCuda();
}
template<size_t activeSize>
void NetworkCudaTexture<activeSize>::randomizeMatrixErrors() {
	weightsE.enableCuda();
	float* r = new float[activeSize * activeSize];
	for (uint32_t u = 0u; u < activeSize * activeSize; u++) r[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	cudaMemcpy(weightsE.data, r, activeSize * activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] r;
	weightsE.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::displace() {
	displaceValues();
	displaceErrors();
	displaceMatrixValues();
	displaceMatrixErrors();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::displaceValues() {
	values.enableCuda();
	float* rh = new float[activeSize];
	for (uint32_t u = 0u; u < activeSize; u++) rh[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	float* rd = nullptr;
	cudaMalloc((void**)&rd, activeSize * sizeof(float));
	cudaMemcpy(rd, rh, activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] rh;
	if (activeSize > 1024u) displaceData<<<(((activeSize - 1u) >> 10u) + 1u), 1024u>>>(values.data, rd, activeSize);
	else if (activeSize > 512u) displaceData <<<1u, 1024u>>>(values.data, rd, activeSize);
	else if (activeSize > 256u) displaceData <<<1u, 512u>>>(values.data, rd, activeSize);
	else if (activeSize > 128u) displaceData <<<1u, 256u>>>(values.data, rd, activeSize);
	else if (activeSize > 64u) displaceData <<<1u, 128u>>>(values.data, rd, activeSize);
	else if (activeSize > 32u) displaceData <<<1u, 64u>>>(values.data, rd, activeSize);
	else displaceData<<<1u, 32u>>>(values.data, rd, activeSize);
	cudaFree(rd);
	values.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::displaceErrors() {
	errors.enableCuda();
	float* rh = new float[activeSize];
	for (uint32_t u = 0u; u < activeSize; u++) rh[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	float* rd = nullptr;
	cudaMalloc((void**)&rd, activeSize * sizeof(float));
	cudaMemcpy(rd, rh, activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] rh;
	if (activeSize > 1024u) displaceData << <(((activeSize - 1u) >> 10u) + 1u), 1024u >> > (errors.data, rd, activeSize);
	else if (activeSize > 512u) displaceData << <1u, 1024u >> > (errors.data, rd, activeSize);
	else if (activeSize > 256u) displaceData << <1u, 512u >> > (errors.data, rd, activeSize);
	else if (activeSize > 128u) displaceData << <1u, 256u >> > (errors.data, rd, activeSize);
	else if (activeSize > 64u) displaceData << <1u, 128u >> > (errors.data, rd, activeSize);
	else if (activeSize > 32u) displaceData << <1u, 64u >> > (errors.data, rd, activeSize);
	else displaceData << <1u, 32u >> > (errors.data, rd, activeSize);
	cudaFree(rd);
	errors.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::displaceMatrixValues() {
	weightsV.enableCuda();
	float* rh = new float[activeSize];
	for (uint32_t u = 0u; u < activeSize; u++) rh[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	float* rd = nullptr;
	cudaMalloc((void**)&rd, activeSize * sizeof(float));
	cudaMemcpy(rd, rh, activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] rh;
	displaceData<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u>>>(weightsV.data, rd, activeSize * activeSize);
	cudaFree(rd);
	weightsV.disableCuda();
}
template<size_t activeSize>
void NetworkCudaTexture<activeSize>::displaceMatrixErrors() {
	weightsE.enableCuda();
	float* rh = new float[activeSize];
	for (uint32_t u = 0u; u < activeSize; u++) rh[u] = ((2.0f * ((float)std::rand() / (float)RAND_MAX)) - 1.0f) * 0.001f;
	float* rd = nullptr;
	cudaMalloc((void**)&rd, activeSize * sizeof(float));
	cudaMemcpy(rd, rh, activeSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] rh;
	displaceData<<<((((activeSize * activeSize) - 1u) >> 10u) + 1u), 1024u >> > (weightsE.data, rd, activeSize * activeSize);
	cudaFree(rd);
	weightsE.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::train() {
	
	calcErrors();
	zeroNValues();
	calcNormAF(values.data, valuesN);
	calcWeights();
	calcValues();
	zeroNErrors();
	calcNorm(errors.data, errorsN);

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
void NetworkCudaTexture<activeSize>::enableCudaValues() {
	values.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::enableCudaErrors() {
	errors.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::enableCudaWeights() {
	weightsV.enableCuda();
	weightsE.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::disableCudaValues() {
	values.disableCuda();
}
template<size_t activeSize>
void NetworkCudaTexture<activeSize>::disableCudaErrors() {
	errors.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::disableCudaWeights() {
	weightsV.disableCuda();
	weightsE.disableCuda();
}

template<size_t activeSize>
float4 NetworkCudaTexture<activeSize>::getAABB() const {
	return aabb;
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::setAABB(float x, float y, float width, float height)
{
	float w = 0.5f * width, h = 0.5f * height;
	values.setRect(x, y + h, w, h);
	errors.setRect(x + w, y + h, w, h);
	weightsE.setRect(x + w, y, w, h);
	weightsV.setRect(x, y, w, h);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::setAABB(float4 ab) {
	float w = 0.5f * ab.z, h = 0.5f * ab.w;
	values.setRect(ab.x, ab.y + h, w, h);
	errors.setRect(ab.x + w, ab.y + h, w, h);
	weightsE.setRect(ab.x + w, ab.y, w, h);
	weightsV.setRect(ab.x, ab.y, w, h);
}


template<size_t activeSize>
void NetworkCudaTexture<activeSize>::enableCuda() {
	errors.enableCuda();
	values.enableCuda();
	weightsV.enableCuda();
	weightsE.enableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::disableCuda() {
	errors.disableCuda();
	values.disableCuda();
	weightsV.disableCuda();
	weightsE.disableCuda();
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::setValues(uint32_t index, uint32_t size, float* data) {
	if (index + size <= activeSize) cudaMemcpy(values.data + index, data, size * sizeof(float), cudaMemcpyHostToDevice);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::getValues(uint32_t index, uint32_t size, float* data) {
	if (index + size <= activeSize) cudaMemcpy(data, values.data + index, size * sizeof(float), cudaMemcpyDeviceToHost);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::setErrors(uint32_t index, uint32_t size, float* data) {
	if (index + size <= activeSize) cudaMemcpy(errors.data + index, data, size * sizeof(float), cudaMemcpyHostToDevice);
}

template<size_t activeSize>
void NetworkCudaTexture<activeSize>::getErrors(uint32_t index, uint32_t size, float* data) {
	if (index + size <= activeSize) cudaMemcpy(data, errors.data + index, size * sizeof(float), cudaMemcpyDeviceToHost);
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