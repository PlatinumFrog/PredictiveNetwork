#pragma once
#include "vector.cuh"

struct matrix2 { float2 x, y; };
struct matrix3 { float3 x, y, z; };
struct matrix4 { float4 x, y, z, w; };


__host__ inline void print(matrix2 matrixA) { std::cout << "\n{" << matrixA.x.x << ", " << matrixA.y.x << "}\n{" << matrixA.x.y << ", " << matrixA.y.y << "}\n"; };
__host__ __device__ inline matrix2 operator+(const matrix2 matrixA, const matrix2 matrixB) { return matrix2{matrixA.x + matrixB.x, matrixA.y + matrixB.y}; };
__host__ __device__ inline matrix2 operator-(const matrix2 matrixA, const matrix2 matrixB) { return matrix2{matrixA.x - matrixB.x, matrixA.y - matrixB.y}; };
__host__ __device__ inline matrix2 operator*(const matrix2 matrixA, const float scalarB) { return matrix2{matrixA.x * scalarB, matrixA.y * scalarB}; };
__host__ __device__ inline matrix2 operator*(const float scalarA, const matrix2 matrixB) { return matrix2{scalarA * matrixB.x, scalarA * matrixB.y}; };
__host__ __device__ inline float2 operator*(const matrix2 matrixA, const float2 vectorB) { return (matrixA.x * vectorB.x) + (matrixA.y * vectorB.y); };
__host__ __device__ inline matrix2 operator/(const matrix2 matrixA, const float scalarB) { return matrix2{matrixA.x / scalarB, matrixA.y / scalarB}; };
__host__ __device__ inline matrix2 operator*(const matrix2 matrixA, const matrix2 matrixB) {
	return matrix2{
		float2{
			(matrixA.x.x * matrixB.x.x) + (matrixA.y.x * matrixB.x.y),
			(matrixA.x.x * matrixB.y.x) + (matrixA.y.x * matrixB.y.y)
		},
		float2{
			(matrixA.x.y * matrixB.x.x) + (matrixA.y.y * matrixB.x.y),
			(matrixA.x.y * matrixB.y.x) + (matrixA.y.y * matrixB.y.y)
		}
	};
};

__host__ __device__ inline matrix2 trn(const matrix2 matrixA) {
	return matrix2{
		float2{matrixA.x.x, matrixA.y.x},
		float2{matrixA.x.y, matrixA.y.y}
	};
};
__host__ __device__ inline float det(const matrix2 matrixA) {
	return (matrixA.x.x * matrixA.y.y) - (matrixA.x.y * matrixA.y.x);
};
__host__ __device__ inline matrix2 inv(const matrix2 matrixA) {
	return matrix2{
		float2{matrixA.y.y, -matrixA.x.y},
		float2{-matrixA.y.x, matrixA.x.x},
	} / ((matrixA.x.x * matrixA.y.y) - (matrixA.x.y * matrixA.y.x));
};


__host__ inline void print(matrix3 matrixA) { std::cout << "\n{" << matrixA.x.x << ", " << matrixA.y.x << ", " << matrixA.z.x << "}\n{" << matrixA.x.y << ", " << matrixA.y.y << ", " << matrixA.z.y << "}\n{" << matrixA.x.z << ", " << matrixA.y.z << ", " << matrixA.z.z << "}\n"; };
__host__ __device__ inline matrix3 operator+(const matrix3 matrixA, const matrix3 matrixB) { return {matrixA.x + matrixB.x, matrixA.y + matrixB.y, matrixA.z + matrixB.z}; };
__host__ __device__ inline matrix3 operator-(const matrix3 matrixA, const matrix3 matrixB) { return {matrixA.x - matrixB.x, matrixA.y - matrixB.y, matrixA.z - matrixB.z}; };
__host__ __device__ inline matrix3 operator*(const matrix3 matrixA, const float scalarB) { return {matrixA.x * scalarB, matrixA.y * scalarB, matrixA.z * scalarB}; };
__host__ __device__ inline matrix3 operator*(const float scalarA, const matrix3 matrixB) { return {scalarA * matrixB.x, scalarA * matrixB.y, scalarA * matrixB.z}; };
__host__ __device__ inline float3 operator*(const matrix3 matrixA, const float3 vectorB) { return (matrixA.x * vectorB.x) + (matrixA.y * vectorB.y) + (matrixA.z * vectorB.z); };
__host__ __device__ inline matrix3 operator/(const matrix3 matrixA, const float scalarB) { return {matrixA.x / scalarB, matrixA.y / scalarB, matrixA.z / scalarB}; };
__host__ __device__ inline matrix3 operator*(const matrix3 matrixA, const matrix3 matrixB) {
	return matrix3{
		float3{
			(matrixA.x.x * matrixB.x.x) + (matrixA.y.x * matrixB.x.y) + (matrixA.z.x * matrixB.x.z),
			(matrixA.x.y * matrixB.x.x) + (matrixA.y.y * matrixB.x.y) + (matrixA.z.y * matrixB.x.z),
			(matrixA.x.z * matrixB.x.x) + (matrixA.y.z * matrixB.x.y) + (matrixA.z.z * matrixB.x.z)
		},
		float3{
			(matrixA.x.x * matrixB.y.x) + (matrixA.y.x * matrixB.y.y) + (matrixA.z.x * matrixB.y.z),
			(matrixA.x.y * matrixB.y.x) + (matrixA.y.y * matrixB.y.y) + (matrixA.z.y * matrixB.y.z),
			(matrixA.x.z * matrixB.y.x) + (matrixA.y.z * matrixB.y.y) + (matrixA.z.z * matrixB.y.z)
		},
		float3{
			(matrixA.x.x * matrixB.z.x) + (matrixA.y.x * matrixB.z.y) + (matrixA.z.x * matrixB.z.z),
			(matrixA.x.y * matrixB.z.x) + (matrixA.y.y * matrixB.z.y) + (matrixA.z.y * matrixB.z.z),
			(matrixA.x.z * matrixB.z.x) + (matrixA.y.z * matrixB.z.y) + (matrixA.z.z * matrixB.z.z)
		}
	};
};
__host__ __device__ inline matrix3 trn(const matrix3 matrixA) {
	return matrix3{
		float3{matrixA.x.x, matrixA.y.x, matrixA.z.x},
		float3{matrixA.x.y, matrixA.y.y, matrixA.z.y},
		float3{matrixA.x.z, matrixA.y.z, matrixA.z.z}
	};
};
__host__ __device__ inline float det(const matrix3 matrixA) {
	return (matrixA.x.x * ((matrixA.y.y * matrixA.z.z) - (matrixA.y.z * matrixA.z.y))) +
		   (matrixA.x.y * ((matrixA.y.z * matrixA.z.x) - (matrixA.y.x * matrixA.z.z))) +
		   (matrixA.x.z * ((matrixA.y.x * matrixA.z.y) - (matrixA.y.y * matrixA.z.x)));
};
__host__ __device__ inline matrix3 inv(const matrix3 matrixA) {
	float3 detE = {
		(matrixA.y.y * matrixA.z.z) - (matrixA.z.y * matrixA.y.z),
		(matrixA.z.y * matrixA.x.z) - (matrixA.x.y * matrixA.z.z),
		(matrixA.x.y * matrixA.y.z) - (matrixA.y.y * matrixA.x.z)
	};
	return matrix3{
		detE,
		float3{
			(matrixA.z.x * matrixA.y.z) - (matrixA.y.x * matrixA.z.z),
			(matrixA.x.x * matrixA.z.z) - (matrixA.z.x * matrixA.x.z),
			(matrixA.y.x * matrixA.x.z) - (matrixA.x.x * matrixA.y.z)
		},
		float3{
			(matrixA.y.x * matrixA.z.y) - (matrixA.z.x * matrixA.y.y),
			(matrixA.z.x * matrixA.x.y) - (matrixA.x.x * matrixA.z.y),
			(matrixA.x.x * matrixA.y.y) - (matrixA.y.x * matrixA.x.y)
		}
	} / ((matrixA.x.x * detE.x) + (matrixA.y.x * detE.y) + (matrixA.z.x * detE.x));
};

__host__ inline void print(matrix4 matrixA) { std::cout << "\n{" << matrixA.x.x << ", " << matrixA.y.x << ", " << matrixA.z.x << ", " << matrixA.w.x << "}\n{" << matrixA.x.y << ", " << matrixA.y.y << ", " << matrixA.z.y << ", " << matrixA.w.y << "}\n{" << matrixA.x.z << ", " << matrixA.y.z << ", " << matrixA.z.z << ", " << matrixA.w.z << "}\n{" << matrixA.x.w << ", " << matrixA.y.w << ", " << matrixA.z.w << ", " << matrixA.w.w << "}\n"; };
__host__ __device__ inline matrix4 operator+(const matrix4 matrixA, const matrix4 matrixB) { return {matrixA.x + matrixB.x, matrixA.y + matrixB.y, matrixA.z + matrixB.z, matrixA.w + matrixB.w}; };
__host__ __device__ inline matrix4 operator-(const matrix4 matrixA, const matrix4 matrixB) { return {matrixA.x - matrixB.x, matrixA.y - matrixB.y, matrixA.z - matrixB.z, matrixA.w - matrixB.w}; };
__host__ __device__ inline matrix4 operator*(const matrix4 matrixA, const float scalarB) { return {matrixA.x * scalarB, matrixA.y * scalarB, matrixA.z * scalarB, matrixA.w * scalarB}; };
__host__ __device__ inline matrix4 operator*(const float scalarA, const matrix4 matrixB) { return {scalarA * matrixB.x, scalarA * matrixB.y, scalarA * matrixB.z, scalarA * matrixB.w}; };
__host__ __device__ inline float4 operator*(const matrix4 matrixA, const float4 vectorB) { return (matrixA.x * vectorB.x) + (matrixA.y * vectorB.y) + (matrixA.z * vectorB.z) + (matrixA.w * vectorB.w); };
__host__ __device__ inline matrix4 operator/(const matrix4 matrixA, const float scalarB) { return {matrixA.x / scalarB, matrixA.y / scalarB, matrixA.z / scalarB, matrixA.w / scalarB}; };
__host__ __device__ inline matrix4 operator*(const matrix4 matrixA, const matrix4 matrixB) {
	return matrix4{
		float4{
			(matrixA.x.x * matrixB.x.x) + (matrixA.y.x * matrixB.x.y) + (matrixA.z.x * matrixB.x.z) + (matrixA.w.x * matrixB.x.w),
			(matrixA.x.y * matrixB.x.x) + (matrixA.y.y * matrixB.x.y) + (matrixA.z.y * matrixB.x.z) + (matrixA.w.y * matrixB.x.w),
			(matrixA.x.z * matrixB.x.x) + (matrixA.y.z * matrixB.x.y) + (matrixA.z.z * matrixB.x.z) + (matrixA.w.z * matrixB.x.w),
			(matrixA.x.w * matrixB.x.x) + (matrixA.y.w * matrixB.x.y) + (matrixA.z.w * matrixB.x.z) + (matrixA.w.w * matrixB.x.w)
		},
		float4{
			(matrixA.x.x * matrixB.y.x) + (matrixA.y.x * matrixB.y.y) + (matrixA.z.x * matrixB.y.z) + (matrixA.w.x * matrixB.y.w),
			(matrixA.x.y * matrixB.y.x) + (matrixA.y.y * matrixB.y.y) + (matrixA.z.y * matrixB.y.z) + (matrixA.w.y * matrixB.y.w),
			(matrixA.x.z * matrixB.y.x) + (matrixA.y.z * matrixB.y.y) + (matrixA.z.z * matrixB.y.z) + (matrixA.w.z * matrixB.y.w),
			(matrixA.x.w * matrixB.y.x) + (matrixA.y.w * matrixB.y.y) + (matrixA.z.w * matrixB.y.z) + (matrixA.w.w * matrixB.y.w)
		},
		float4{
			(matrixA.x.x * matrixB.z.x) + (matrixA.y.x * matrixB.z.y) + (matrixA.z.x * matrixB.z.z) + (matrixA.w.x * matrixB.z.w),
			(matrixA.x.y * matrixB.z.x) + (matrixA.y.y * matrixB.z.y) + (matrixA.z.y * matrixB.z.z) + (matrixA.w.y * matrixB.z.w),
			(matrixA.x.z * matrixB.z.x) + (matrixA.y.z * matrixB.z.y) + (matrixA.z.z * matrixB.z.z) + (matrixA.w.z * matrixB.z.w),
			(matrixA.x.w * matrixB.z.x) + (matrixA.y.w * matrixB.z.y) + (matrixA.z.w * matrixB.z.z) + (matrixA.w.w * matrixB.z.w)
		},
		float4{
			(matrixA.x.x * matrixB.w.x) + (matrixA.y.x * matrixB.w.y) + (matrixA.z.x * matrixB.w.z) + (matrixA.w.x * matrixB.w.w),
			(matrixA.x.y * matrixB.w.x) + (matrixA.y.y * matrixB.w.y) + (matrixA.z.y * matrixB.w.z) + (matrixA.w.y * matrixB.w.w),
			(matrixA.x.z * matrixB.w.x) + (matrixA.y.z * matrixB.w.y) + (matrixA.z.z * matrixB.w.z) + (matrixA.w.z * matrixB.w.w),
			(matrixA.x.w * matrixB.w.x) + (matrixA.y.w * matrixB.w.y) + (matrixA.z.w * matrixB.w.z) + (matrixA.w.w * matrixB.w.w)
		}
	};
};
__host__ __device__ inline matrix4 trn(const matrix4 matrixA) {
	return matrix4{
		float4{matrixA.x.x, matrixA.y.x, matrixA.z.x, matrixA.w.x},
		float4{matrixA.x.y, matrixA.y.y, matrixA.z.y, matrixA.w.y},
		float4{matrixA.x.z, matrixA.y.z, matrixA.z.z, matrixA.w.z},
		float4{matrixA.x.w, matrixA.y.w, matrixA.z.w, matrixA.w.w}
	};
};
__host__ __device__ inline float det(const matrix4 matrixA) {
	float d[6] = {
		(matrixA.z.z * matrixA.w.w) - (matrixA.z.w * matrixA.w.z),
		(matrixA.z.x * matrixA.w.y) - (matrixA.z.y * matrixA.w.x),
		(matrixA.z.y * matrixA.w.z) - (matrixA.z.z * matrixA.w.y),
		(matrixA.z.w * matrixA.w.x) - (matrixA.z.x * matrixA.w.w),
		(matrixA.z.w * matrixA.w.y) - (matrixA.z.y * matrixA.w.w),
		(matrixA.z.z * matrixA.w.x) - (matrixA.z.x * matrixA.w.z)
	};
	return
		  matrixA.x.x * ((matrixA.y.z * d[4]) + (matrixA.y.y * d[0]) + (matrixA.y.w * d[2]))
		+ matrixA.x.y * ((matrixA.y.w * d[5]) - (matrixA.y.z * d[3]) - (matrixA.y.x * d[0]))
		+ matrixA.x.z * ((matrixA.y.y * d[3]) + (matrixA.y.w * d[1]) - (matrixA.y.x * d[4]))
		- matrixA.x.w * ((matrixA.y.z * d[1]) + (matrixA.y.x * d[2]) + (matrixA.y.y * d[5]));
};

__host__ __device__ inline matrix4 inv(const matrix4 matrixA) {
	float d[20] = {
		(matrixA.z.z * matrixA.w.w) - (matrixA.w.z * matrixA.z.w), //0
		(matrixA.x.z * matrixA.y.w) - (matrixA.y.z * matrixA.x.w), //1
		(matrixA.y.z * matrixA.z.w) - (matrixA.z.z * matrixA.y.w), //2
		(matrixA.w.z * matrixA.x.w) - (matrixA.x.z * matrixA.w.w), //3
		(matrixA.z.y * matrixA.w.w) - (matrixA.w.y * matrixA.z.w), //4
		(matrixA.x.y * matrixA.y.w) - (matrixA.y.y * matrixA.x.w), //5
		(matrixA.y.y * matrixA.z.w) - (matrixA.z.y * matrixA.y.w), //6
		(matrixA.w.y * matrixA.x.w) - (matrixA.x.y * matrixA.w.w), //7
		(matrixA.z.y * matrixA.w.z) - (matrixA.w.y * matrixA.z.z), //8
		(matrixA.x.y * matrixA.y.z) - (matrixA.y.y * matrixA.x.z), //9
		(matrixA.y.y * matrixA.z.z) - (matrixA.z.y * matrixA.y.z), //10
		(matrixA.w.y * matrixA.x.z) - (matrixA.x.y * matrixA.w.z), //11
		(matrixA.w.z * matrixA.y.w) - (matrixA.y.z * matrixA.w.w), //12
		(matrixA.z.z * matrixA.x.w) - (matrixA.x.z * matrixA.z.w), //13
		(matrixA.w.y * matrixA.y.w) - (matrixA.y.y * matrixA.w.w), //16
		(matrixA.z.y * matrixA.x.w) - (matrixA.x.y * matrixA.z.w), //17
		(matrixA.w.y * matrixA.y.z) - (matrixA.y.y * matrixA.w.z), //18
		(matrixA.z.y * matrixA.x.z) - (matrixA.x.y * matrixA.z.z)  //19
	};
	matrix4 M = matrix4{
		float4{ (matrixA.y.y * d[0]) + (matrixA.w.y * d[2]) + (matrixA.z.y * d[12]),
				(matrixA.w.y * d[13]) - (matrixA.z.y * d[3]) - (matrixA.x.y * d[0]),
				(matrixA.w.y * d[1]) + (matrixA.y.y * d[3]) - (matrixA.x.y * d[12]),
			   -(matrixA.z.y * d[1]) - (matrixA.x.y * d[2]) - (matrixA.y.y * d[13])},
		float4{-(matrixA.y.x * d[0]) - (matrixA.w.x * d[2]) - (matrixA.z.x * d[12]),
				(matrixA.x.x * d[0]) + (matrixA.z.x * d[3]) - (matrixA.w.x * d[13]),
				(matrixA.x.x * d[12]) - (matrixA.w.x * d[1]) - (matrixA.y.x * d[3]),
				(matrixA.z.x * d[1]) + (matrixA.x.x * d[2]) + (matrixA.y.x * d[13])},
		float4{ (matrixA.y.x * d[4]) + (matrixA.w.x * d[6]) + (matrixA.z.x * d[14]),
				(matrixA.w.x * d[15]) - (matrixA.z.x * d[7]) - (matrixA.x.x * d[4]),
				(matrixA.w.x * d[5]) + (matrixA.y.x * d[7]) - (matrixA.x.x * d[14]),
			   -(matrixA.z.x * d[5]) - (matrixA.x.x * d[6]) - (matrixA.y.x * d[15])},
		float4{-(matrixA.y.x * d[8]) - (matrixA.w.x * d[10]) - (matrixA.z.x * d[16]),
				(matrixA.x.x * d[8]) + (matrixA.z.x * d[11]) - (matrixA.w.x * d[17]),
				(matrixA.x.x * d[16]) - (matrixA.w.x * d[9]) - (matrixA.y.x * d[11]),
				(matrixA.z.x * d[9]) + (matrixA.x.x * d[10]) + (matrixA.y.x * d[17])}
	};
	return M * (1.0f/((matrixA.x.x * M.x.x) + (matrixA.y.x * M.x.y) + (matrixA.z.x * M.x.z) + (matrixA.w.x * M.x.w)));
};

__host__ __device__ inline matrix4 cinv(const matrix4 matrixA) {
	float d[9] = {
		(matrixA.y.y * matrixA.z.z) - (matrixA.z.y * matrixA.y.z),
		(matrixA.z.y * matrixA.x.z) - (matrixA.x.y * matrixA.z.z),
		(matrixA.x.y * matrixA.y.z) - (matrixA.y.y * matrixA.x.z),

		(matrixA.z.x * matrixA.y.z) - (matrixA.y.x * matrixA.z.z),
		(matrixA.x.x * matrixA.z.z) - (matrixA.z.x * matrixA.x.z),
		(matrixA.y.x * matrixA.x.z) - (matrixA.x.x * matrixA.y.z),

		(matrixA.y.x * matrixA.z.y) - (matrixA.z.x * matrixA.y.y),
		(matrixA.z.x * matrixA.x.y) - (matrixA.x.x * matrixA.z.y),
		(matrixA.x.x * matrixA.y.y) - (matrixA.y.x * matrixA.x.y)
	};
	float det = 1.0f/(matrixA.x.x * d[0] + matrixA.x.y * d[3] + matrixA.x.z * d[6]);
	return matrix4{
		float4{d[0] * det,
			   d[1] * det,
			   d[2] * det,
			   0.0f
		},
		float4{d[3] * det,
			   d[4] * det,
			   d[5] * det,
			   0.0f
		},
		float4{d[6] * det,
			   d[7] * det,
			   d[8] * det,
			   0.0f
		},
		float4{(-(matrixA.w.x * d[0]) - (matrixA.w.y * d[3]) - (matrixA.w.z * d[6])) * det,
			   (-(matrixA.w.x * d[1]) - (matrixA.w.y * d[4]) - (matrixA.w.z * d[7])) * det,
			   (-(matrixA.w.x * d[2]) - (matrixA.w.y * d[5]) - (matrixA.w.z * d[8])) * det,
			   1.0f
		}
	};

	

};