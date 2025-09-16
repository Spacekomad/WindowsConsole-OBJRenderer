#pragma once


class Light {
public:
	Light() {
		rgb[0] = 200;
		rgb[1] = 220;
		rgb[2] = 255;
	}

public:
	unsigned char rgb[3];
};