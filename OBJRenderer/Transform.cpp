#include "Transform.h"

Transform::Transform()
{
	_modelingMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
	_worldSpaceNormalTransform = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
	_scale.x = 1.0f;
	_scale.y = 1.0f;
	_scale.z = 1.0f;
}

void Transform::SetPosition(float x, float y, float z)
{
	_position.x = x;
	_position.y = y;
	_position.z = z;
}

void Transform::SetPosition(Vector3f position)
{
	_position.x = position.x;
	_position.y = position.y;
	_position.z = position.z;
}

void Transform::SetRotation(float x, float y, float z)
{
	if (x > 360.0f) {
		x = std::fmod(x, 360);
	}
	if (x < 0.0f) {
		x *= -1.0f;
		x = 360 - std::fmod(x, 360);
	}
	if (y > 360.0f) {
		y = std::fmod(y, 360);
	}
	if (y < 0.0f) {
		y *= -1.0f;
		y = 360 - std::fmod(y, 360);
	}
	if (z > 360.0f) {
		z = std::fmod(z, 360);
	}
	if (z < 0.0f) {
		z *= -1.0f;
		z = 360 - std::fmod(z, 360);
	}

	_rotation.x = x;
	_rotation.y = y;
	_rotation.z = z;
}

void Transform::SetRotation(Vector3f rotation)
{
	SetRotation(rotation.x, rotation.y, rotation.z);
}

void Transform::SetScale(float x, float y, float z)
{
	_scale.x = x;
	_scale.y = y;
	_scale.z = z;
}

void Transform::SetScale(Vector3f scale)
{
	_scale.x = scale.x;
	_scale.y = scale.y;
	_scale.z = scale.z;
}

const Vector3f& Transform::GetPosition() const
{
	return _position;
}

const Vector3f& Transform::GetRotation() const
{
	return _rotation;
}

const Vector3f& Transform::GetScale() const
{
	return _scale;
}

const cv::Mat& Transform::GetModelingMatrix()
{
	float sx = sin(_rotation.x * DEGREE_TO_RADIAN);
	float sy = sin(_rotation.y * DEGREE_TO_RADIAN);
	float sz = sin(_rotation.z * DEGREE_TO_RADIAN);
	float cx = cos(_rotation.x * DEGREE_TO_RADIAN);
	float cy = cos(_rotation.y * DEGREE_TO_RADIAN);
	float cz = cos(_rotation.z * DEGREE_TO_RADIAN);

	_modelingMatrix.at<float>(0, 3) = _position.x;
	_modelingMatrix.at<float>(1, 3) = _position.y;
	_modelingMatrix.at<float>(2, 3) = _position.z;

	_modelingMatrix.at<float>(0, 0) = _scale.x * cy * cz;
	_modelingMatrix.at<float>(0, 1) =  -1.0f * _scale.y * cy * sz;
	_modelingMatrix.at<float>(0, 2) = _scale.z * sy;

	_modelingMatrix.at<float>(1, 0) = _scale.x * (sx * sy * cz + cx * sz);
	_modelingMatrix.at<float>(1, 1) = _scale.y * (cx * cz - sx * sy * sz);
	_modelingMatrix.at<float>(1, 2) = -1.0f * _scale.z * sx * cy;

	_modelingMatrix.at<float>(2, 0) = _scale.x * (sx * sz - cx * sy * cz);
	_modelingMatrix.at<float>(2, 1) = _scale.y * (cx * sy * sz + sx * cz);
	_modelingMatrix.at<float>(2, 2) = _scale.z * cx * cy;

	_modelingMatrix.at<float>(3, 3) = 1.0f;

	/*std::cout << "\nModeling\n";
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::cout << _modelingMatrix.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/

	return _modelingMatrix;
}

cv::Mat& Transform::GetWorldSpaceNormalTransform()
{
	_worldSpaceNormalTransform.at<float>(3, 3) = 1.0f;

	float sx = sin(_rotation.x * DEGREE_TO_RADIAN);
	float sy = sin(_rotation.y * DEGREE_TO_RADIAN);
	float sz = sin(_rotation.z * DEGREE_TO_RADIAN);
	float cx = cos(_rotation.x * DEGREE_TO_RADIAN);
	float cy = cos(_rotation.x * DEGREE_TO_RADIAN);
	float cz = cos(_rotation.x * DEGREE_TO_RADIAN);

	_worldSpaceNormalTransform.at<float>(0, 0) = cy * cz;
	_worldSpaceNormalTransform.at<float>(0, 1) = -1.0f * cy * sz;
	_worldSpaceNormalTransform.at<float>(0, 2) = sy;

	_worldSpaceNormalTransform.at<float>(1, 0) = sx * sy * cz + cx * sz;
	_worldSpaceNormalTransform.at<float>(1, 1) = cx * cz - sx * sy * sz;
	_worldSpaceNormalTransform.at<float>(1, 2) = -1.0f * sx * cy;

	_worldSpaceNormalTransform.at<float>(2, 0) = sx * sz - cx * sy * cz;
	_worldSpaceNormalTransform.at<float>(2, 1) = cx * sy * sz + sx * cz;
	_worldSpaceNormalTransform.at<float>(2, 2) = cx * cy;

	return _worldSpaceNormalTransform;
}
