#pragma once

#include <vector>

#include "GameObject.h"
#include "Camera.h"
#include "DirectionalLight.h"
#include "Color.h"

class Scene {
public:
	Scene();

#pragma region Variables
public:
	Color skyColor;
	DirectionalLight mainLight;
	std::shared_ptr<Camera> mainCamera;
	std::vector<std::shared_ptr<Light>> lights;
	std::vector<std::shared_ptr<GameObject>> gameObjects;

private:

#pragma endregion
#pragma region Functions
public:
	std::shared_ptr<GameObject> CreateGameObject();
	std::shared_ptr<GameObject> CreateGameObject(std::string modelFileLocation);
	std::shared_ptr<GameObject> CreateGameObject(std::string modelFileLocation, std::string textureFileLocation);

	void SetMainLightDirection(float x, float y, float z);
	void AddLight(std::shared_ptr<Light>& light);

private:

#pragma endregion

};