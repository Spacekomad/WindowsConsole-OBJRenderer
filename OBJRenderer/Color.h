#pragma once


class Color {
public:
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;

	float GetFloatR() {
		return r / 255;
	}
	float GetFloatG() {
		return g / 255;
	}
	float GetFloatB() {
		return b / 255;
	}
	float GetFloatAlpha() {
		return a / 255;
	}
};