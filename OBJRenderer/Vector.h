#pragma once

#include <cmath>

struct Vector4f {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 0.0f;

	Vector4f() {

	}
	Vector4f(float x, float y, float z, float w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
};

struct Vector3f {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	Vector3f() {

	}
	Vector3f(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vector3f& operator-(Vector3f& other) {
		this->x -= other.x;
		this->y -= other.y;
		this->z -= other.z;
		return *this;
	}
	Vector3f& operator+(Vector3f& other) {
		this->x += other.x;
		this->y += other.y;
		this->z += other.z;
		return *this;
	}
	Vector3f& operator/(float value) {
		this->x /= value;
		this->y /= value;
		this->z /= value;
		return *this;
	}
	Vector3f& operator*(float value) {
		this->x *= value;
		this->y *= value;
		this->z *= value;
		return *this;
	}

	void Normalize() {
		if (this->x != 0.0f && this->y != 0.0f && this->z != 0.0f) {
			float length = sqrtf(this->x * this->x + this->y * this->y + this->z * this->z);
			this->x /= length;
			this->y /= length;
			this->z /= length;
		}
	}
};

struct Vector3i {
	int x = 0;
	int y = 0;
	int z = 0;
	int w = 0;
};

struct Vector4i {
	int x = 0;
	int y = 0;
	int z = 0;
	int w = 0;

	Vector4i() { }

	Vector4i(int x, int y, int z, int w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
};