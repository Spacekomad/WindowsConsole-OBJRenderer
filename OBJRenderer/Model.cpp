#include "Model.h"

void Model::SetMem()
{
	vertexPosition.reserve(vertexNumber);
	vertexNormal.reserve(vertexNormalNumber);
	vertexUV.reserve(uvNumber);
	facePositionIndex.reserve(faceNumber);
	faceUVIndex.reserve(faceNumber);
	faceNormalIndex.reserve(faceNumber);
}

void Model::PrintModelInfoToText()
{
	std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
	for (Vector4f vertex : vertexPosition) {
		std::cout << "(" << vertex.x << ", " << vertex.y << ", " << vertex.z << ", " << vertex.w << ") ";
		std::cout << "\n";
	}

	std::cout << "ttttttttttttttttttttttttttttttt\n";
	for (Vector3f vertex : vertexUV) {
		std::cout << "(" << vertex.x << ", " << vertex.y << ", " << vertex.z << ") ";
		std::cout << "\n";
	}
	std::cout << "\n";


	std::cout << "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn\n";
	for (Vector3f vertex : vertexNormal) {
		std::cout << "(" << vertex.x << ", " << vertex.y << ", " << vertex.z << ") ";
		std::cout << "\n";
	}
	std::cout << "\n";

	std::cout << "ffffffffffffffffffffffffffffffff\n";
	for (int i = 0; i < faceNumber; ++i) {
		std::cout << "(" << facePositionIndex[i].x << ", " << facePositionIndex[i].y << ", " << facePositionIndex[i].z << ") ";
		std::cout << "\n";
		std::cout << "(" << faceUVIndex[i].x << ", " << faceUVIndex[i].y << ", " << faceUVIndex[i].z << ") ";
		std::cout << "\n";
		std::cout << "(" << faceNormalIndex[i].x << ", " << faceNormalIndex[i].y << ", " << faceNormalIndex[i].z << ") ";
		std::cout << "\n";
		std::cout << "\n";
	}
	std::cout << "\n";
}

std::shared_ptr<Model> Model::LoadModel(const std::string fileLocation)
{
	std::ifstream file;
	file.open(fileLocation);
	assert(file.is_open());

	std::shared_ptr<Model> newModel = std::make_shared<Model>();

	// 버텍스와 페이스 개수 카운트
	while (!file.eof()) {
		std::string fileInput;

		std::getline(file, fileInput);

		if (fileInput[0] == 'v' && fileInput[1] == ' ')
			++newModel->vertexNumber;
		else if (fileInput[0] == 'v' && fileInput[1] == 't')
			++newModel->uvNumber;
		else if (fileInput[0] == 'v' && fileInput[1] == 'n')
			++newModel->vertexNormalNumber;
		else if (fileInput[0] == 'f' && fileInput[1] == ' ')
			++newModel->faceNumber;

		fileInput.clear();
	}
	file.close();

	// 메모리 세팅 후
	newModel->SetMem();

	file.open(fileLocation);
	assert(file.is_open());

	int vertexIndex = 0;
	int faceIndex = 0;
	int uvIndex = 0;
	int vertextNoramlIndex = 0;

	// 버텍스 정보 기입
	while (!file.eof()) {
		float value;
		std::string fileInput;

		file >> fileInput;

		if (fileInput.compare("v") == 0) {
			Vector4f newVector4;

			file >> value;
			newVector4.x = value;

			file >> value;
			newVector4.y = value;

			file >> value;
			newVector4.z = value;

			file >> value;
			if (!file.fail()) {
				newVector4.w = value;
			}
			else {
				newVector4.w = 1.0f;
				file.clear();
			}

			newModel->vertexPosition.push_back(newVector4);

			++vertexIndex;
		}
		else if (fileInput.compare("vt") == 0) {
			Vector3f newVector3;

			file >> value;
			newVector3.x = value;

			file >> value;
			newVector3.y = value;

			file >> value;
			if (!file.fail()) {
				newVector3.z = value;
			}
			else {
				file.clear();
			}

			newModel->vertexUV.push_back(newVector3);

			++uvIndex;
		}
		else if (fileInput.compare("vn") == 0) {
			Vector3f newVector3;

			file >> value;
			newVector3.x = value;

			file >> value;
			newVector3.y = value;

			file >> value;
			if (!file.fail()) {
				newVector3.z = value;
			}
			else {
				file.clear();
			}

			newModel->vertexNormal.push_back(newVector3);

			++vertextNoramlIndex;
		}
		else if (fileInput.compare("f") == 0) {
			Vector3i newVector[4]; // x엔 position, y엔 uv, z엔 normal, index는 face를 구성하는 몇 번째 버텍스인가
			char delimiter;
			for (int i = 0; i < 3; ++i) {
				file >> value;
				newVector[i].x = value - 1;
				file >> delimiter;

				file >> value;
				if (!file.fail()) {
					newVector[i].y = value - 1;
				}
				else {
					newVector[i].y = -1;
					file.clear();
				}

				file >> delimiter;
				file >> value;
				newVector[i].z = value - 1;
			}

			file >> value;

			if (!file.fail()) {// 쿼드인 경우 삼각형 분할해서 페이스 하나 더 추가
				Vector3i newFace[3];
				newFace[0].x = newVector[2].x;
				newFace[0].y = newVector[2].y;
				newFace[0].z = newVector[2].z;

				newFace[2].x = newVector[0].x;
				newFace[2].y = newVector[0].y;
				newFace[2].z = newVector[0].z;

				newFace[1].x = value - 1;
				file >> delimiter;

				file >> value;
				if (!file.fail()) {
					newFace[1].y = value - 1;
				}
				else {
					newFace[1].y = -1;
					file.clear();
				}

				file >> delimiter;
				file >> value;
				newFace[1].z = value - 1;

				newModel->facePositionIndex.push_back(Vector4i(newFace[0].x, newFace[1].x, newFace[2].x, -1));
				newModel->faceUVIndex.push_back(Vector4i(newFace[0].y, newFace[1].y, newFace[2].y, -1));
				newModel->faceNormalIndex.push_back(Vector4i(newFace[0].z, newFace[1].z, newFace[2].z, -1));

				++newModel->faceNumber;
			}
			else {// 삼각형인 경우 
				newVector[3].x = -1;
				newVector[3].y = -1;
				newVector[3].z = -1;
				file.clear();
			}

			newModel->facePositionIndex.push_back(Vector4i(newVector[0].x, newVector[1].x, newVector[2].x, newVector[3].x));
			newModel->faceUVIndex.push_back(Vector4i(newVector[0].y, newVector[1].y, newVector[2].y, newVector[3].y));
			newModel->faceNormalIndex.push_back(Vector4i(newVector[0].z, newVector[1].z, newVector[2].z, newVector[3].z));

			++faceIndex;
		}
		fileInput.clear();
	}
	file.close();

	return newModel;
}

std::shared_ptr<Model> Model::LoadModel(const std::string fileLocation, const std::string textureLocation)
{
	cv::Mat srcImg = cv::imread(textureLocation);

	std::shared_ptr<Model>& newModel = LoadModel(fileLocation);

	newModel->texture = srcImg;

	return newModel;
}
