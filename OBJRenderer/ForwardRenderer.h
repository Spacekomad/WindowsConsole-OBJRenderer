#pragma once

#include "RenderFunctions.h"

#include "Renderer.h"

class ForwardRenderer : public Renderer
{
public:

#pragma region Variables
public:

protected:

private:

#pragma endregion
#pragma region Functions
public:
	const std::vector<uchar>& RenderScene(const Scene& targetScene) override;

protected:

private:

#pragma endregion
};