#pragma once

#include <vector>

#include "Scene.h"

class Renderer
{
public:
#pragma region Variables
public:

protected:

private:

#pragma endregion
#pragma region Functions
public:
	virtual const std::vector<uchar>& RenderScene(const Scene& targetScene) = 0;

protected:

private:

#pragma endregion

};