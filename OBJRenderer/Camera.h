#pragma once

#include <vector>

#include "Transform.h"

class Camera {
public:
	Camera();

#pragma region Variables
public:
	float cameraNear = 0.3f;
	float cameraFar = 100.0f;

	Transform transform;

private:
	int _frameWidth = 300;
	int _frameHeight = 100;
	float _aspectRatio;
	float _fieldOfView = 60; // vertical °¢µµ
	std::vector<uchar> _frameBuffer;

#pragma endregion
#pragma region Functions
public:
	int GetFrameBufferSize() const;
	int GetFrameSize() const;
	int GetFrameWidth() const;
	int GetFrameHeight() const;
	std::vector<uchar>& GetFrameBuffer();
	cv::Mat& GetViewportMatrix();
	const cv::Mat& GetViewingMatrix() ;
	const cv::Mat& GetPerspectiveProjectionMatrix() ;
	const cv::Mat& GetVPMatrix() ;

private:
	cv::Mat _viewportMatrix;
	cv::Mat _viewingMatrix;
	cv::Mat _perspectiveProjectionMatrix;
	cv::Mat _VPMatrix;
	cv::Mat _supportMatrix;

#pragma endregion
};