#pragma once

#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <windows.h>
#include <opencv2/opencv.hpp>

#include "Renderer.h"

#pragma region PreprocessorDefine

#define PIXEL_BLACK 0
#define PIXEL_WHITE 15

#define PIXEL_BLUE 1
#define PIXEL_GREEN 2
#define PIXEL_RED 4

#define PIXEL_INTENSITY FOREGROUND_INTENSITY

#pragma endregion

class ConsoleMonitor
{
public:
	struct RGBColor
	{
	private:
		uchar intensity = 0;
	public:
		int color = 0;
		void SetIntensity(int intensity) {
			if (intensity < 0) {
				intensity = 0;
			}
			if (intensity > 255) {
				intensity = 255;
			}
			this->intensity = intensity / 37;
		}
		uchar GetIntensity() const {
			return intensity;
		}
	};
	struct Pixel
	{
		RGBColor rgb[3];

		Pixel() {
			rgb[0].color = PIXEL_RED;
			rgb[1].color = PIXEL_GREEN;
			rgb[2].color = PIXEL_BLUE;
		}
	};

public:
	ConsoleMonitor(const std::shared_ptr<Renderer>& renderer, 
		const int consoleWidth, const int consoleHeight, 
		const int displayWidth, const int displayHeight, 
		const int frameBufferSize = 2);

	~ConsoleMonitor();

#pragma region Variables
public:
	const char* pixelText = " ";
	const int pixelTextLength = 1;
	int consoleBackgroundColor;
	int consoleWidth;
	int consoleHeight;
	int displayWidth;
	int displayHeight;

	int frameBufferIndex = 0;
	std::vector<std::vector<std::vector<Pixel>>> pixels;

	bool colorFlag;
	CONSOLE_FONT_INFOEX cfi;

	std::shared_ptr<Renderer> renderer;

private:
	HANDLE _hStdOut;

#pragma endregion
#pragma region Functions
public:
	void InitializeConsole();
	void ResizeConsole(const int consoleWidth, const int consoleHeight);
	void DrawDisplay();
	void DrawDisplayGray();
	void SetImageToPixels(std::string imgLocation);
	void SetCameraBufferToPixels(const std::shared_ptr<Camera>& frameBuffer);
	void RenderScene(const Scene& targetScene);

private:
	inline void SetPixelColor(int color);
	inline void SetTextColor(int foreground, int background);
	void SetConsoleGray();
	void SetConsoleColor();

#pragma endregion

};
