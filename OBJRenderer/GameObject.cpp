#include "GameObject.h"

GameObject::GameObject()
{
	model.resize(0);
}

std::shared_ptr<GameObject> GameObject::CreateGameObject()
{
	return std::shared_ptr<GameObject>(new GameObject());// GameObject 积己磊 private
}

std::shared_ptr<GameObject> GameObject::CreateGameObject(std::string modelFileLocation)
{
	GameObject go;
	go.model.push_back(Model::LoadModel(modelFileLocation));
	return std::make_shared<GameObject>(go); // 汗荤 积己磊 磊悼 积己
}

std::shared_ptr<GameObject> GameObject::CreateGameObject(std::string modelFileLocation, std::string textureFileLocation)
{
	GameObject go;
	go.model.push_back(Model::LoadModel(modelFileLocation, textureFileLocation));
	return std::make_shared<GameObject>(go); // 汗荤 积己磊 磊悼 积己
}
