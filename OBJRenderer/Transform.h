#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>

#include "Vector.h"

#define PI 3.141592
#define DEGREE_TO_RADIAN 0.0174533
#define RADIAN_TODEGREE 57.2958

class Transform
{
public:
	Transform();

#pragma region Variables
public:

private:
	Vector3f _position;
	Vector3f _rotation;
	Vector3f _scale;

#pragma endregion
#pragma region Functions
public:
	void SetPosition(float x, float y, float z);
	void SetPosition(Vector3f position);
	void SetRotation(float x, float y, float z);
	void SetRotation(Vector3f rotation);
	void SetScale(float x, float y, float z);
	void SetScale(Vector3f scale);
	const Vector3f& GetPosition() const;
	const Vector3f& GetRotation() const;
	const Vector3f& GetScale() const;
	const cv::Mat& GetModelingMatrix();
	cv::Mat& GetWorldSpaceNormalTransform();

private:
	cv::Mat _modelingMatrix;
	cv::Mat _worldSpaceNormalTransform;

#pragma endregion

};