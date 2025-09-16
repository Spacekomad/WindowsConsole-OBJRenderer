#include "Scene.h"

Scene::Scene()
{
    mainLight.direction.x = 1.0f;
    mainLight.direction.y = -1.0f;
    mainLight.direction.z = 1.0f;
    mainLight.direction.Normalize();

    skyColor.r = 0;
    skyColor.g = 0;
    skyColor.b = 0;
    mainCamera = std::make_shared<Camera>();
    lights.resize(0);
    gameObjects.resize(0);
}

std::shared_ptr<GameObject> Scene::CreateGameObject()
{
    std::shared_ptr<GameObject> gameObject = GameObject::CreateGameObject();
    gameObjects.push_back(gameObject);
    return gameObject;
}

std::shared_ptr<GameObject> Scene::CreateGameObject(std::string modelFileLocation)
{
    std::shared_ptr<GameObject> gameObject = GameObject::CreateGameObject(modelFileLocation);
    gameObjects.push_back(gameObject);
    return gameObject;
}

std::shared_ptr<GameObject> Scene::CreateGameObject(std::string modelFileLocation, std::string textureFileLocation)
{
    std::shared_ptr<GameObject> gameObject = GameObject::CreateGameObject(modelFileLocation, textureFileLocation);
    gameObjects.push_back(gameObject);
    return gameObject;
}

void Scene::SetMainLightDirection(float x, float y, float z)
{
    mainLight.direction.x = x;
    mainLight.direction.y = y;
    mainLight.direction.z = z;
    mainLight.direction.Normalize();
}

void Scene::AddLight(std::shared_ptr<Light>& light)
{
    lights.push_back(light);
}
