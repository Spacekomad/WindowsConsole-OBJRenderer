#pragma once

#include <vector>

#include "Model.h"
#include "Transform.h"

class Model;

class GameObject
{
private:
	// �׳� ������ �� ��. heap���� �ø� ����.
	GameObject();

#pragma region Variables
public:
	bool active = true;

	std::vector<std::shared_ptr<Model>> model;
	Transform transform;

private:

#pragma endregion
#pragma region Functions
public:
	static std::shared_ptr<GameObject> CreateGameObject();
	static std::shared_ptr<GameObject> CreateGameObject(std::string modelFileLocation);
	static std::shared_ptr<GameObject> CreateGameObject(std::string modelFileLocation, std::string textureFileLocation);

private:


#pragma endregion

};
