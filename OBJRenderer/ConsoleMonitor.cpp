#include "ConsoleMonitor.h"

ConsoleMonitor::ConsoleMonitor(const std::shared_ptr<Renderer>& renderer, 
	const int consoleWidth, const int consoleHeight, 
	const int displayWidth, const int displayHeight, 
	const int frameBufferSize)
{
	this->renderer = renderer;

	colorFlag = true;
	consoleBackgroundColor = PIXEL_BLACK;

	this->consoleWidth = consoleWidth;
	this->consoleHeight = consoleHeight;
	this->displayHeight = displayHeight;
	this->displayWidth = displayWidth;

	pixels.resize(frameBufferSize, std::vector<std::vector<Pixel>>(displayHeight, std::vector<Pixel>(displayWidth)));
	InitializeConsole();
}

ConsoleMonitor::~ConsoleMonitor()
{
	CloseHandle(_hStdOut);
}

void ConsoleMonitor::InitializeConsole()
{
	_hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);

	// 글자 색을 rgb 픽셀로 쓰기 위해 크기 지정
	cfi.cbSize = sizeof cfi;
	cfi.nFont = 0;
	cfi.dwFontSize.X = 0;
	cfi.dwFontSize.Y = 2;
	cfi.FontFamily = FF_DONTCARE;
	cfi.FontWeight = FW_NORMAL;
	std::wcscpy(cfi.FaceName, L"OBJ Pixel");
	SetCurrentConsoleFontEx(_hStdOut, FALSE, &cfi);

	// 콘솔 창 크기 지정
	ResizeConsole(consoleWidth, consoleHeight);
	//system("mode con: cols=900 lines=300");
}

void ConsoleMonitor::ResizeConsole(const int consoleWidth, const int consoleHeight)
{
	std::string sizeString = "mode con: cols=" + std::to_string(consoleWidth) + " lines=" + std::to_string(consoleHeight);
	system(sizeString.c_str());
}

// 픽셀정보 화면 출력
void ConsoleMonitor::DrawDisplay()
{
	SetConsoleColor();

	system("cls");

	for (int y = 0; y < displayHeight; ++y) {
		for (int consolY = 0; consolY < 3; ++consolY) { // 3라인 찍기 (rgb 3x3이 한 픽셀)
			for (int x = 0; x < displayWidth; ++x) { // 한 라인 찍기
				for (int k = 0; k < 3; ++k) { // rgb 하나 찍기
					int color = pixels[frameBufferIndex][y][x].rgb[k].GetIntensity() + (2 - consolY);

					switch (color / 3)
					{
					case 0:
						SetPixelColor(consoleBackgroundColor);
						break;
					case 1:
						SetPixelColor(pixels[frameBufferIndex][y][x].rgb[k].color);
						break;
					case 2:
						SetPixelColor(pixels[frameBufferIndex][y][x].rgb[k].color | PIXEL_INTENSITY);
						break;
					default:
						SetPixelColor(consoleBackgroundColor);
						break;
					}
					fwrite(pixelText, sizeof(char), pixelTextLength, stdout);
				}
			}
		}
	}
}

void ConsoleMonitor::DrawDisplayGray()
{
	SetConsoleGray();

	system("cls");

	for (int y = 0; y < displayHeight; ++y) {
		for (int x = 0; x < displayWidth; ++x) {
			int gray = 0;
			for (int k = 0; k < 3; ++k) {
				gray += pixels[frameBufferIndex][y][x].rgb[k].GetIntensity();
			}

			switch ((gray + 2) / 6) {
			case 0:
				SetPixelColor(0);
				break;
			case 1:
				SetPixelColor(8);
				break;
			case 2:
				SetPixelColor(7);
				break;
			case 3:
				SetPixelColor(15);
				break;
			default:
				break;
			}

			fwrite(pixelText, sizeof(char), pixelTextLength, stdout);
		}
	}
}

void ConsoleMonitor::SetImageToPixels(std::string imgLocation) {
	cv::Mat srcImg = cv::imread(imgLocation);

	int i = 0;
	for (int y = 0; y < displayHeight; ++y) {
		uchar* rows = srcImg.ptr<uchar>(y);
		for (int x = 0; x < displayWidth; ++x) {
			pixels[frameBufferIndex][y][x].rgb[0].SetIntensity(rows[x * 3 + 2]);
			pixels[frameBufferIndex][y][x].rgb[1].SetIntensity(rows[x * 3 + 1]);
			pixels[frameBufferIndex][y][x].rgb[2].SetIntensity(rows[x * 3]);

			++i;
		}
	}
}

void ConsoleMonitor::SetCameraBufferToPixels(const std::shared_ptr<Camera>& camera)
{
	int xIter = camera->GetFrameWidth() < displayWidth ? camera->GetFrameWidth() : displayWidth;
	int yIter = camera->GetFrameHeight() < displayHeight ? camera->GetFrameHeight() : displayHeight;

	std::vector<uchar>& frameBuffer = camera->GetFrameBuffer();

	int i = 0;
	for (int y = 0; y < yIter; ++y) {
		for (int x = 0; x < xIter; ++x) {
			pixels[frameBufferIndex][y][x].rgb[0].SetIntensity(frameBuffer[y * 3 * xIter + x * 3]);
			pixels[frameBufferIndex][y][x].rgb[1].SetIntensity(frameBuffer[y * 3 * xIter + x * 3 + 1]);
			pixels[frameBufferIndex][y][x].rgb[2].SetIntensity(frameBuffer[y * 3 * xIter + x * 3 + 2]);

			++i;
		}
	}
}

void ConsoleMonitor::RenderScene(const Scene& targetScene)
{
	renderer->RenderScene(targetScene);
	SetCameraBufferToPixels(targetScene.mainCamera);
	DrawDisplay();
}


inline void ConsoleMonitor::SetPixelColor(int color)
{
	SetTextColor(consoleBackgroundColor, color);
}

inline void ConsoleMonitor::SetTextColor(int foreground, int background) {
	int color = foreground | (background << 4);

	SetConsoleTextAttribute(_hStdOut, color);
}

void ConsoleMonitor::SetConsoleGray()
{
	if (colorFlag) {
		colorFlag = false;

		cfi.dwFontSize.Y = 5;
		SetCurrentConsoleFontEx(_hStdOut, FALSE, &cfi);
		ResizeConsole(consoleWidth / 3, consoleHeight / 3);
	}
}

void ConsoleMonitor::SetConsoleColor()
{
	if (!colorFlag) {
		colorFlag = true;

		cfi.dwFontSize.Y = 2;
		SetCurrentConsoleFontEx(_hStdOut, FALSE, &cfi);
		ResizeConsole(consoleWidth, consoleHeight);
	}
}

