#pragma once
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>



__host__ inline void print(float2 vectorA) { std::cout << "\n{" << vectorA.x << ", " << vectorA.y << "}\n"; };
__host__ __device__ inline float2 operator+(const float scalarA, const float2 vectorB) { return {(scalarA + vectorB.x), (scalarA + vectorB.y)}; };
__host__ __device__ inline float2 operator+(const float2 vectorA, const float scalarB) { return {(vectorA.x + scalarB), (vectorA.y + scalarB)}; };
__host__ __device__ inline float2 operator+(const float2 vectorA, const float2 vectorB) { return {(vectorA.x + vectorB.x), (vectorA.y + vectorB.y)}; };
__host__ __device__ inline float2 operator-(const float scalarA, const float2 vectorB) { return {(scalarA - vectorB.x), (scalarA - vectorB.y)}; };
__host__ __device__ inline float2 operator-(const float2 vectorA, const float scalarB) { return {(vectorA.x - scalarB), (vectorA.y - scalarB)}; };
__host__ __device__ inline float2 operator-(const float2 vectorA, const float2 vectorB) { return {(vectorA.x - vectorB.x), (vectorA.y - vectorB.y)}; };
__host__ __device__ inline float2 operator*(const float scalarA, const float2 vectorB) { return {(scalarA * vectorB.x), (scalarA * vectorB.y)}; };
__host__ __device__ inline float2 operator*(const float2 vectorA, const float scalarB) { return {(vectorA.x * scalarB), (vectorA.y * scalarB)}; };
__host__ __device__ inline float2 operator*(const float2 vectorA, const float2 vectorB) { return {(vectorA.x * vectorB.x), (vectorA.y * vectorB.y)}; };
__host__ __device__ inline float2 operator/(const float scalarA, const float2 vectorB) { return {(scalarA / vectorB.x), (scalarA / vectorB.y)}; };
__host__ __device__ inline float2 operator/(const float2 vectorA, const float scalarB) { return {(vectorA.x / scalarB), (vectorA.y / scalarB)}; };
__host__ __device__ inline float2 operator/(const float2 vectorA, const float2 vectorB) { return {(vectorA.x / vectorB.x), (vectorA.y / vectorB.y)}; };
__host__ __device__ inline void operator+=(float2& vectorA, const float scalarB) { vectorA = {(vectorA.x + scalarB), (vectorA.y + scalarB)}; };
__host__ __device__ inline void operator+=(float2& vectorA, const float2 vectorB) { vectorA = {(vectorA.x + vectorB.x), (vectorA.y + vectorB.y)}; };
__host__ __device__ inline void operator-=(float2& vectorA, const float scalarB) { vectorA = {(vectorA.x - scalarB), (vectorA.y - scalarB)}; };
__host__ __device__ inline void operator-=(float2& vectorA, const float2 vectorB) { vectorA = {(vectorA.x - vectorB.x), (vectorA.y - vectorB.y)}; };
__host__ __device__ inline void operator*=(float2& vectorA, const float scalarB) { vectorA = {(vectorA.x * scalarB), (vectorA.y * scalarB)}; };
__host__ __device__ inline void operator*=(float2& vectorA, const float2 vectorB) { vectorA = {(vectorA.x * vectorB.x), (vectorA.y * vectorB.y)}; };
__host__ __device__ inline void operator/=(float2& vectorA, const float scalarB) { vectorA = {(vectorA.x / scalarB), (vectorA.y / scalarB)}; };
__host__ __device__ inline void operator/=(float2& vectorA, const float2 vectorB) { vectorA = {(vectorA.x / vectorB.x), (vectorA.y / vectorB.y)}; };
__host__ __device__ inline float2 operator-(const float2 vectorA) { return {-vectorA.x, -vectorA.y}; };
__host__ __device__ inline float dot(const float2 vectorA, const float2 vectorB) { return (vectorA.x * vectorB.x) + (vectorA.y * vectorB.y); };
__host__ __device__ inline float2 norm(const float2 vectorA) { return vectorA / sqrt(dot(vectorA, vectorA)); };

__host__ inline void print(float3 vectorA) { std::cout << "\n{" << vectorA.x << ", " << vectorA.y << ", " << vectorA.z << "}\n"; };
__host__ __device__ inline float3 operator+(const float scalarA, const float3 vectorB) { return {(scalarA + vectorB.x), (scalarA + vectorB.y), (scalarA + vectorB.z)}; };
__host__ __device__ inline float3 operator+(const float3 vectorA, const float scalarB) { return {(vectorA.x + scalarB), (vectorA.y + scalarB), (vectorA.z + scalarB)}; };
__host__ __device__ inline float3 operator+(const float3 vectorA, const float3 vectorB) { return {(vectorA.x + vectorB.x), (vectorA.y + vectorB.y), (vectorA.z + vectorB.z)}; };
__host__ __device__ inline float3 operator-(const float scalarA, const float3 vectorB) { return {(scalarA - vectorB.x), (scalarA - vectorB.y), (scalarA - vectorB.z)}; };
__host__ __device__ inline float3 operator-(const float3 vectorA, const float scalarB) { return {(vectorA.x - scalarB), (vectorA.y - scalarB), (vectorA.z - scalarB)}; };
__host__ __device__ inline float3 operator-(const float3 vectorA, const float3 vectorB) { return {(vectorA.x - vectorB.x), (vectorA.y - vectorB.y), (vectorA.z - vectorB.z)}; };
__host__ __device__ inline float3 operator*(const float scalarA, const float3 vectorB) { return {(scalarA * vectorB.x), (scalarA * vectorB.y), (scalarA * vectorB.z)}; };
__host__ __device__ inline float3 operator*(const float3 vectorA, const float scalarB) { return {(vectorA.x * scalarB), (vectorA.y * scalarB), (vectorA.z * scalarB)}; };
__host__ __device__ inline float3 operator*(const float3 vectorA, const float3 vectorB) { return {(vectorA.x * vectorB.x), (vectorA.y * vectorB.y), (vectorA.z * vectorB.z)}; };
__host__ __device__ inline float3 operator/(const float scalarA, const float3 vectorB) { return {(scalarA / vectorB.x), (scalarA / vectorB.y), (scalarA / vectorB.z)}; };
__host__ __device__ inline float3 operator/(const float3 vectorA, const float scalarB) { return {(vectorA.x / scalarB), (vectorA.y / scalarB), (vectorA.z / scalarB)}; };
__host__ __device__ inline float3 operator/(const float3 vectorA, const float3 vectorB) { return {(vectorA.x / vectorB.x), (vectorA.y / vectorB.y), (vectorA.z / vectorB.z)}; };
__host__ __device__ inline void operator+=(float3& vectorA, const float scalarB) { vectorA = {(vectorA.x + scalarB), (vectorA.y + scalarB), (vectorA.z + scalarB)}; };
__host__ __device__ inline void operator+=(float3& vectorA, const float3 vectorB) { vectorA = {(vectorA.x + vectorB.x), (vectorA.y + vectorB.y), (vectorA.z + vectorB.z)}; };
__host__ __device__ inline void operator-=(float3& vectorA, const float scalarB) { vectorA = {(vectorA.x - scalarB), (vectorA.y - scalarB), (vectorA.z - scalarB)}; };
__host__ __device__ inline void operator-=(float3& vectorA, const float3 vectorB) { vectorA = {(vectorA.x - vectorB.x), (vectorA.y - vectorB.y), (vectorA.z - vectorB.z)}; };
__host__ __device__ inline void operator*=(float3& vectorA, const float scalarB) { vectorA = {(vectorA.x * scalarB), (vectorA.y * scalarB), (vectorA.z * scalarB)}; };
__host__ __device__ inline void operator*=(float3& vectorA, const float3 vectorB) { vectorA = {(vectorA.x * vectorB.x), (vectorA.y * vectorB.y), (vectorA.z * vectorB.z)}; };
__host__ __device__ inline void operator/=(float3& vectorA, const float scalarB) { vectorA = {(vectorA.x / scalarB), (vectorA.y / scalarB), (vectorA.z / scalarB)}; };
__host__ __device__ inline void operator/=(float3& vectorA, const float3 vectorB) { vectorA = {(vectorA.x / vectorB.x), (vectorA.y / vectorB.y), (vectorA.z / vectorB.z)}; };
__host__ __device__ inline float3 operator-(const float3 vectorA) { return {-vectorA.x, -vectorA.y, -vectorA.z}; };
__host__ __device__ inline float dot(const float3 vectorA, const float3 vectorB) { return (vectorA.x * vectorB.x) + (vectorA.y * vectorB.y) + (vectorA.z * vectorB.z); };
__host__ __device__ inline float3 cross(const float3 vectorA, const float3 vectorB) { return {(vectorA.y * vectorB.z) - (vectorA.z * vectorB.y), (vectorA.z * vectorB.x) - (vectorA.x * vectorB.z), (vectorA.x * vectorB.y) - (vectorA.y * vectorB.x)}; };
__host__ __device__ inline float3 norm(const float3 vectorA) { return vectorA / sqrt(dot(vectorA, vectorA)); };

__host__ inline void print(float4 vectorA) { std::cout << "\n{" << vectorA.x << ", " << vectorA.y << ", " << vectorA.z << ", " << vectorA.w << "}\n"; };
__host__ __device__ inline float4 operator+(const float scalarA, const float4 vectorB) { return {(scalarA + vectorB.x), (scalarA + vectorB.y), (scalarA + vectorB.z), (scalarA + vectorB.w)}; };
__host__ __device__ inline float4 operator+(const float4 vectorA, const float scalarB) { return {(vectorA.x + scalarB), (vectorA.y + scalarB), (vectorA.z + scalarB), (vectorA.w + scalarB)}; };
__host__ __device__ inline float4 operator+(const float4 vectorA, const float4 vectorB) { return {(vectorA.x + vectorB.x), (vectorA.y + vectorB.y), (vectorA.z + vectorB.z), (vectorA.w + vectorB.w)}; };
__host__ __device__ inline float4 operator-(const float scalarA, const float4 vectorB) { return {(scalarA - vectorB.x), (scalarA - vectorB.y), (scalarA - vectorB.z), (scalarA - vectorB.w)}; };
__host__ __device__ inline float4 operator-(const float4 vectorA, const float scalarB) { return {(vectorA.x - scalarB), (vectorA.y - scalarB), (vectorA.z - scalarB), (vectorA.w - scalarB)}; };
__host__ __device__ inline float4 operator-(const float4 vectorA, const float4 vectorB) { return {(vectorA.x - vectorB.x), (vectorA.y - vectorB.y), (vectorA.z - vectorB.z), (vectorA.w - vectorB.w)}; };
__host__ __device__ inline float4 operator*(const float scalarA, const float4 vectorB) { return {(scalarA * vectorB.x), (scalarA * vectorB.y), (scalarA * vectorB.z), (scalarA * vectorB.w)}; };
__host__ __device__ inline float4 operator*(const float4 vectorA, const float scalarB) { return {(vectorA.x * scalarB), (vectorA.y * scalarB), (vectorA.z * scalarB), (vectorA.w * scalarB)}; };
__host__ __device__ inline float4 operator*(const float4 vectorA, const float4 vectorB) { return {(vectorA.x * vectorB.x), (vectorA.y * vectorB.y), (vectorA.z * vectorB.z), (vectorA.w * vectorB.w)}; };
__host__ __device__ inline float4 operator/(const float scalarA, const float4 vectorB) { return {(scalarA / vectorB.x), (scalarA / vectorB.y), (scalarA / vectorB.z), (scalarA / vectorB.w)}; };
__host__ __device__ inline float4 operator/(const float4 vectorA, const float scalarB) { return {(vectorA.x / scalarB), (vectorA.y / scalarB), (vectorA.z / scalarB), (vectorA.w / scalarB)}; };
__host__ __device__ inline float4 operator/(const float4 vectorA, const float4 vectorB) { return {(vectorA.x / vectorB.x), (vectorA.y / vectorB.y), (vectorA.z / vectorB.z), (vectorA.w / vectorB.w)}; };
__host__ __device__ inline void operator+=(float4& vectorA, const float scalarB) { vectorA = {(vectorA.x + scalarB), (vectorA.y + scalarB), (vectorA.z + scalarB), (vectorA.w + scalarB)}; };
__host__ __device__ inline void operator+=(float4& vectorA, const float4 vectorB) { vectorA = {(vectorA.x + vectorB.x), (vectorA.y + vectorB.y), (vectorA.z + vectorB.z), (vectorA.w + vectorB.w)}; };
__host__ __device__ inline void operator-=(float4& vectorA, const float scalarB) { vectorA = {(vectorA.x - scalarB), (vectorA.y - scalarB), (vectorA.z - scalarB), (vectorA.w - scalarB)}; };
__host__ __device__ inline void operator-=(float4& vectorA, const float4 vectorB) { vectorA = {(vectorA.x - vectorB.x), (vectorA.y - vectorB.y), (vectorA.z - vectorB.z), (vectorA.w - vectorB.w)}; };
__host__ __device__ inline void operator*=(float4& vectorA, const float scalarB) { vectorA = {(vectorA.x * scalarB), (vectorA.y * scalarB), (vectorA.z * scalarB), (vectorA.w * scalarB)}; };
__host__ __device__ inline void operator*=(float4& vectorA, const float4 vectorB) { vectorA = {(vectorA.x * vectorB.x), (vectorA.y * vectorB.y), (vectorA.z * vectorB.z), (vectorA.w * vectorB.w)}; };
__host__ __device__ inline void operator/=(float4& vectorA, const float scalarB) { vectorA = {(vectorA.x / scalarB), (vectorA.y / scalarB), (vectorA.z / scalarB), (vectorA.w / scalarB)}; };
__host__ __device__ inline void operator/=(float4& vectorA, const float4 vectorB) { vectorA = {(vectorA.x / vectorB.x), (vectorA.y / vectorB.y), (vectorA.z / vectorB.z), (vectorA.w / vectorB.w)}; };
__host__ __device__ inline float4 operator-(const float4 vectorA) { return {-vectorA.x, -vectorA.y, -vectorA.z, -vectorA.w}; };
__host__ __device__ inline float dot(const float4 vectorA, const float4 vectorB) { return (vectorA.x * vectorB.x) + (vectorA.y * vectorB.y) + (vectorA.z * vectorB.z) + (vectorA.w * vectorB.w); };
__host__ __device__ inline float4 norm(const float4 vectorA) { return vectorA / sqrt(dot(vectorA, vectorA)); };
