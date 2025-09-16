#include "Camera.h"

Camera::Camera()
{
	transform.SetPosition(0, 10, 30);
	transform.SetRotation(0, 0, 0);
	_aspectRatio = _frameWidth / _frameHeight;
	_frameBuffer.resize(_frameWidth * _frameHeight * 3);

	_supportMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
	_viewportMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
	_viewingMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
	_perspectiveProjectionMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
	_VPMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);
}

int Camera::GetFrameBufferSize() const
{
	return _frameBuffer.size();
}

int Camera::GetFrameSize() const
{
	return _frameWidth * _frameHeight;
}

int Camera::GetFrameWidth() const
{
	return _frameWidth;
}

int Camera::GetFrameHeight() const
{
	return _frameHeight;
}

std::vector<uchar>& Camera::GetFrameBuffer()
{
	return _frameBuffer;
}

cv::Mat& Camera::GetViewportMatrix()
{

	_viewportMatrix.at<float>(0, 0) = 0.5f * _frameWidth;
	_viewportMatrix.at<float>(1, 1) = 0.5f * _frameHeight;
	_viewportMatrix.at<float>(2, 2) = 0.5f;

	_viewportMatrix.at<float>(0, 3) = 0.5f * _frameWidth;
	_viewportMatrix.at<float>(1, 3) = 0.5f * _frameHeight;
	_viewportMatrix.at<float>(2, 3) = 0.5f;

	_viewportMatrix.at<float>(3, 3) = 1.0f;

	/*std::cout << "\nViewport\n";
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::cout << _viewportMatrix.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/

	return _viewportMatrix;
}

const cv::Mat& Camera::GetViewingMatrix()
{
	Vector3f position = transform.GetPosition();
	Vector3f rotation = transform.GetRotation();

	float sx = sin(rotation.x * DEGREE_TO_RADIAN);
	float sy = sin(rotation.y * DEGREE_TO_RADIAN);
	float sz = sin(rotation.z * DEGREE_TO_RADIAN);
	float cx = cos(rotation.x * DEGREE_TO_RADIAN);
	float cy = cos(rotation.y * DEGREE_TO_RADIAN);
	float cz = cos(rotation.z * DEGREE_TO_RADIAN);

	// 카메라 좌표축을 기준으로 어파인 변환한 방식
	// 이동변환
	/*_viewingMatrix.at<float>(0, 3) = -1.0f * position.x;
	_viewingMatrix.at<float>(1, 3) = -1.0f * position.y;
	_viewingMatrix.at<float>(2, 3) = -1.0f * position.z;*/

	// 회전변환
	/*_viewingMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);

	_viewingMatrix.at<float>(0, 0) = cy * cz;
	_viewingMatrix.at<float>(0, 1) = cx * sz + sx * sy * cz;
	_viewingMatrix.at<float>(0, 2) = sx * sz - cx * sy * cz;

	_viewingMatrix.at<float>(1, 0) = -cy * sz;
	_viewingMatrix.at<float>(1, 1) = cx * cz - sx * sy * sz;
	_viewingMatrix.at<float>(1, 2) = sx * cz + cx * sy * sz;

	_viewingMatrix.at<float>(2, 0) = sy;
	_viewingMatrix.at<float>(2, 1) = -sx * cy;
	_viewingMatrix.at<float>(2, 2) = cx * cy;

	_viewingMatrix.at<float>(3, 3) = 1.0f;*/

	/*_supportMatrix = cv::Mat::zeros(cv::Size(4, 4), CV_32FC1);

	_supportMatrix.at<float>(0, 3) = -1.0f * position.x;
	_supportMatrix.at<float>(1, 3) = -1.0f * position.y;
	_supportMatrix.at<float>(2, 3) = -1.0f * position.z;

	_supportMatrix.at<float>(0, 0) = 1.0f;
	_supportMatrix.at<float>(1, 1) = 1.0f;
	_supportMatrix.at<float>(2, 2) = 1.0f;
	_supportMatrix.at<float>(3, 3) = 1.0f;

	_viewingMatrix = _viewingMatrix * _supportMatrix;*/

	// 카메라 모델링 변환의 역행렬을 이용한 방식
	_viewingMatrix.at<float>(0, 0) = cy * cz;
	_viewingMatrix.at<float>(0, 1) = cy * sz;
	_viewingMatrix.at<float>(0, 2) = -sy;
	_viewingMatrix.at<float>(0, 3) = position.z * sy - position.y * cy * sz - position.x * cy * cz;

	_viewingMatrix.at<float>(1, 0) = sx * sy * cz - cx * sz;
	_viewingMatrix.at<float>(1, 1) = sx * sy * sz + cx * cz;
	_viewingMatrix.at<float>(1, 2) = sx * cy;
	_viewingMatrix.at<float>(1, 3) = position.z * sx * cy - 
		position.y * (sx * sy * sz + cx * cz) + 
		position.x * (cx * sz - sx * sy * cz);

	_viewingMatrix.at<float>(2, 0) = cx * sy * cz + sx * sz;
	_viewingMatrix.at<float>(2, 1) = cx * sy * sz - sx * cz;
	_viewingMatrix.at<float>(2, 2) = cx * cy;
	_viewingMatrix.at<float>(2, 3) = -position.z * cx * cy -
		position.y * (cx * sy * sz + sx * cz) +
		position.x * (-sx * sz - cx * sy * cz);

	_viewingMatrix.at<float>(3, 3) = 1.0f;

	/*std::cout << "\nViewing\n";
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::cout << _viewingMatrix.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/

	return _viewingMatrix;
}

const cv::Mat& Camera::GetPerspectiveProjectionMatrix()
{
	float cotHalf = 1 / tan(_fieldOfView * 0.5f * DEGREE_TO_RADIAN);
	_perspectiveProjectionMatrix.at<float>(0, 0) = cotHalf / _aspectRatio;
	_perspectiveProjectionMatrix.at<float>(1, 1) = cotHalf;
	_perspectiveProjectionMatrix.at<float>(2, 2) = (cameraNear + cameraFar) / (cameraNear - cameraFar);
	_perspectiveProjectionMatrix.at<float>(2, 3) = 2.0f * cameraNear * cameraFar / (cameraNear - cameraFar);
	_perspectiveProjectionMatrix.at<float>(3, 2) = -1.0f;

	/*std::cout << "\nPerspective\n";
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::cout << _perspectiveProjectionMatrix.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/

	return _perspectiveProjectionMatrix;
}

const cv::Mat& Camera::GetVPMatrix()
{
	_VPMatrix = GetPerspectiveProjectionMatrix() * GetViewingMatrix();

	/*std::cout << "\nVP\n";
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::cout << _VPMatrix.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";*/

	return _VPMatrix;
}
