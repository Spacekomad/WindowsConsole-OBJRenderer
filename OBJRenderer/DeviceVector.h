#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Vector.h"

__device__ struct DEV_Vector4f {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 0.0f;

	__device__ DEV_Vector4f() {

	}
	__device__ DEV_Vector4f(float x, float y, float z, float w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
};

__device__ struct DEV_Vector3f {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	__device__ DEV_Vector3f() {

	}
	__device__ DEV_Vector3f(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__device__ DEV_Vector3f operator-(DEV_Vector3f& other) {
		DEV_Vector3f newVec;
		newVec.x = this->x - other.x;
		newVec.y = this->y - other.y;
		newVec.z = this->z -  other.z;
		return newVec;
	}
	__device__ DEV_Vector3f operator+(DEV_Vector3f& other) {
		DEV_Vector3f newVec;
		newVec.x = this->x + other.x;
		newVec.y = this->y + other.y;
		newVec.z = this->z + other.z;
		return newVec;
	}
	__device__ DEV_Vector3f operator/(float value) {
		DEV_Vector3f newVec;
		newVec.x = this->x / value;
		newVec.y = this->y / value;
		newVec.z = this->z / value;
		return newVec;
	}
	__device__ DEV_Vector3f operator*(float value) {
		DEV_Vector3f newVec;
		newVec.x = this->x * value;
		newVec.y = this->y * value;
		newVec.z = this->z * value;
		return newVec;
	}
	__device__ float operator*(DEV_Vector3f& other) {
		return (this->x * other.x) + (this->y * other.y) + (this->z * other.z);
	}
	__device__ float operator*(Vector3f& other) {
		return (this->x * other.x) + (this->y * other.y) + (this->z * other.z);
	}

	__device__ float GetLength() {
		return sqrtf((this->x * this->x) + (this->y * this->y) + (this->z * this->z));
	}
};

__device__ struct DEV_Vector3i {
	int x = 0;
	int y = 0;
	int z = 0;
	int w = 0;
};

__device__ struct DEV_Vector4i {
	int x = 0;
	int y = 0;
	int z = 0;
	int w = 0;

	__device__ DEV_Vector4i() { }

	__device__ DEV_Vector4i(int x, int y, int z, int w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
};