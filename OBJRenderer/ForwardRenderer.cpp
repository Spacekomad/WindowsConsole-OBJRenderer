#include "ForwardRenderer.h"

const std::vector<uchar>& ForwardRenderer::RenderScene(const Scene& targetScene)
{
	const cv::Mat& VPMat = targetScene.mainCamera->GetVPMatrix();
	int frameWidth = targetScene.mainCamera->GetFrameWidth();
	int frameHeight = targetScene.mainCamera->GetFrameHeight();

	Kernel::MAllocOneFrameElementsToDevice(targetScene.mainCamera->GetFrameSize());
	Kernel::InitializeFrame(targetScene.skyColor, frameWidth, frameHeight);
	Kernel::SetViewportMatrixToDevice(targetScene.mainCamera->GetViewportMatrix());

	for (const auto& gameObject : targetScene.gameObjects) {
		if (gameObject->active) {

			cv::Mat MVP = VPMat * gameObject->transform.GetModelingMatrix();

			/*std::cout << "\nMVP\n";
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					std::cout << MVP.at<float>(i, j) << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";*/

			Kernel::SetMVPToDevice(MVP);
			Kernel::SetWorldSpaceNormalTransform(gameObject->transform.GetWorldSpaceNormalTransform());

			for (const auto& model : gameObject->model) {
				Kernel::SetModelDataToDevice(model);

				// 관측 변환 단계
				Kernel::StartVertexShader(model->vertexNumber, model->vertexNormalNumber);

				// 래스터 라이저 단계
				Kernel::StartRasterizer(model->vertexNumber, model->faceNumber, frameWidth, frameHeight);

				Kernel::StartFragmentShader(targetScene.mainLight, targetScene.skyColor, 
					frameWidth, frameHeight, model->texture.size().width, model->texture.size().height);

				Kernel::DeleteModelMemory();
			}
		}
	}

	Kernel::CopyResultToMonitor(targetScene.mainCamera->GetFrameBuffer().data(), 
		targetScene.mainCamera->GetFrameBufferSize());

	Kernel::DeleteOneFrameElements();

	return targetScene.mainCamera->GetFrameBuffer();
}
