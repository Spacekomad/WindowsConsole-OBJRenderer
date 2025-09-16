#pragma once

#include <vector>

#include "Model.h"
#include "Transform.h"

class Model;

class GameObject
{
private:
	// 그냥 생성은 못 함. heap에만 올릴 예정.
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
