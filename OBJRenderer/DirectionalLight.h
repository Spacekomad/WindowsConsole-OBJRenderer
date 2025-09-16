#pragma once

#include "Vector.h"
#include "Light.h"

class DirectionalLight : public Light
{
public:
	Vector3f direction;
};