#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "DirectionalLight.h"
#include "DeviceVector.h"
#include "Model.h"
#include "Color.h"

#define WARP 32

namespace Kernel {
	__device__ struct Depth {
		float depth;
		DEV_Vector3f uv;
		DEV_Vector3f normal;
	};

#pragma region DeviceFunctions
	__device__ float Clamp(float value, float min, float max);
	__device__ int Clamp(int value, int min, int max);
	__device__ void Swap(DEV_Vector3f* a, DEV_Vector3f* b);
	__device__ void WriteDepth(Depth* depthBuffer, float depth, int depthIndex, 
		int frameSize, int faceIndex, DEV_Vector4i* faceNormalIndex, 
		DEV_Vector3f uv, DEV_Vector3f* worldSpaceNormal);
	__device__ float Lerp(float value, float a, float b);
	__device__ float FastReverseSqrt(float number);
#pragma endregion

#pragma region KernelFunctions
	__global__ void Test(DEV_Vector4i* faceNormalIndex, int faceNumber, DEV_Vector3f* worldSpaceNormal);

	__global__ void Test2(Depth* depthBuffer, int frameWidth, int frameHeight);

	__global__ void WipeFrameBuffer(uchar* frameBuffer, Color skyColor, int frameWidth, int frameHeight);

	__global__ void InitializeDepth(Depth* depthBuffer, int frameWidth, int frameHeight);

	__global__ void VertexShader(DEV_Vector4f* vertexPosition, DEV_Vector4f* clipSpacePosition, 
		float* MVP, int vertexNumber);

	__global__ void LocalToWolrdNormal(DEV_Vector3f* vertextNormal, DEV_Vector3f* worldSpaceNormal, float* WSNT, int normalNumber);

	__global__ void PerspectiveDivision(DEV_Vector4f* clipSpacePosition, int vertexNumber);

	__global__ void BackFaceCulling(DEV_Vector4i* facePositionIndex, DEV_Vector4f* clipSpacePosition, 
		bool* shouldRenderFace, int faceNumber);

	__global__ void ViewportTransform(DEV_Vector4f* clipSpacePosition, DEV_Vector4f* viewportSpacePosition, 
		float* viewportMatrix, int vertexNumber);

	__global__ void ScanConversion(DEV_Vector4i* facePositionIndex, DEV_Vector4i* faceUVIndex, DEV_Vector4i* faceNormalIndex, 
		DEV_Vector4f* viewportSpacePosition, DEV_Vector3f* vertexUV, DEV_Vector3f* worldSpaceNormal, bool* shouldRenderFace,
		Depth* depthBuffer,	int faceNumber, int frameWidth, int frameHeight);

	__global__ void FragmentShader(Depth* depthBuffer, uchar* frameBuffer, uchar* texture, 
		DirectionalLight mainLight, Color skyColor, int frameWidth, int frameHeight, int textureWidth, int textureHeight);
#pragma endregion

#pragma region GPUMemoryMangement
	void MAllocOneFrameElementsToDevice(int frameSize);
	void SetViewportMatrixToDevice(cv::Mat& viewportMatrix);
	void SetMVPToDevice(cv::Mat& MVP);
	void SetWorldSpaceNormalTransform(cv::Mat& WSNT);
	void SetModelDataToDevice(const std::shared_ptr<Model>& model);
	void DeleteModelMemory();
	void DeleteOneFrameElements();
	void CopyResultToMonitor(uchar* camBufferPtr, int frameBufferSize);
#pragma endregion

#pragma region WrapperFunctions
	void InitializeFrame(Color skyColor, int frameWidth, int frameHeight);
	void StartVertexShader(int vertexNumber, int vertexNormalNumber);
	void StartRasterizer(int vertexNumber, int faceNumber, int frameWidth, int frameHeight);
	void StartFragmentShader(DirectionalLight mainLight, Color skyColor, 
		int frameWidth, int frameHeight, int textureWidth, int textureHeight);
#pragma endregion

}
