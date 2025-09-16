#pragma once

#include <vector>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "Vector.h"
#include "AccCtrl.h"

class Model // <- ½Ì±ÛÅæ
{
#pragma region Variables
public:
	int vertexNumber = 0;
	int vertexNormalNumber = 0;
	int uvNumber = 0;
	int faceNumber = 0;

	std::vector<Vector4f> vertexPosition;
	std::vector<Vector3f> vertexUV;
	std::vector<Vector3f> vertexNormal;
	std::vector<Vector4i> facePositionIndex;
	std::vector<Vector4i> faceUVIndex;
	std::vector<Vector4i> faceNormalIndex;
	cv::Mat texture;

private:

#pragma endregion
#pragma region Functions
public:
	void SetMem();
	void PrintModelInfoToText();

	// heap¿¡ ¿Ã¸®°í shared ptr ¹ÝÈ¯
	static std::shared_ptr<Model> LoadModel(const std::string fileLocation);
	static std::shared_ptr<Model> LoadModel(const std::string fileLocation, const std::string textureLocation);

private:

#pragma endregion

};