#include "RenderFunctions.h"

using namespace Kernel;

#pragma region DeviceVariables
__device__ DEV_Vector4f* dev_vertexPosition;
__device__ DEV_Vector3f* dev_vertexUV;
__device__ DEV_Vector3f* dev_vertexNormal;

// �޸� ���縦 1���� �ϴ� �� ���� ������ ������ ������ ��
__device__ DEV_Vector4i* dev_facePositionIndex;
__device__ DEV_Vector4i* dev_faceUVIndex;
__device__ DEV_Vector4i* dev_faceNormalIndex;
__device__ uchar* dev_texture;

__device__ float* dev_MVP;
__device__ float* dev_viewportMatrix;
__device__ float* dev_worldSpaceNormalTransform;

// ����� ���� ����̽� ������
__device__ bool* dev_shouldRenderFace;
__device__ DEV_Vector3f* dev_worldSpaceNormal;
__device__ DEV_Vector4f* dev_clipSpacePosition;
__device__ DEV_Vector4f* dev_viewportSpacePosition;
__device__ Depth* dev_depthBuffer;
// frameBuffer ���� => 0�� �ȼ� r, g, b -> 1�� �ȼ� r, g, b... ���� 0~2���� 1�� �ȼ�
__device__ uchar* dev_frameBuffer;

#pragma endregion
#pragma region DeviceFunctions
__device__ float Kernel::Clamp(float value, float min, float max)
{
	if (value < min) {
		value = min;
	}
	if (value > max) {
		value = max;
	}
	return value;
}
__device__ int Kernel::Clamp(int value, int min, int max)
{
	if (value < min) {
		value = min;
	}
	if (value > max) {
		value = max;
	}
	return value;
}
__device__ void Kernel::Swap(DEV_Vector3f* a, DEV_Vector3f* b)
{
	DEV_Vector3f tmp = *a;
	*a = *b;
	*b = tmp;
}
__device__ void Kernel::WriteDepth(Depth* depthBuffer, float depth, int depthIndex, 
	int frameSize, int faceIndex, DEV_Vector4i* faceNormalIndex, 
	DEV_Vector3f uv, DEV_Vector3f* worldSpaceNormal)
{
	if (depthIndex > frameSize - 1) {
		return;
	}
	// Z test
	if (depth > depthBuffer[depthIndex].depth) {
		return;
	}
	depthBuffer[depthIndex].depth = depth;
	depthBuffer[depthIndex].uv = uv;

	depthBuffer[depthIndex].normal = (worldSpaceNormal[faceNormalIndex[faceIndex].x] +
			worldSpaceNormal[faceNormalIndex[faceIndex].y] + 
			worldSpaceNormal[faceNormalIndex[faceIndex].z]) / 3;
}
__device__ float Kernel::Lerp(float value, float a, float b)
{
	if (value < 0.0f) {
		value = 0.0f;
	}
	if (value > 1.0f) {
		value = 1.0f;
	}

	return value * (b - a) + a;
}

// ��� �� ������
// http://www.matrix67.com/data/InvSqrt.pdf by CHRIS LOMONT
__device__ float Kernel::FastReverseSqrt(float number)
{
	float xhalf = 0.5f * number;
	int i = *(int*)&number; // get bits for floating value
	i = 0x5f375a86 - (i >> 1); // gives initial guess y0
	number = *(float*)&i; // convert bits back to float
	number = number * (1.5f - xhalf * number * number); // Newton step, repeating increases accuracy
	return number;
}

#pragma endregion
#pragma region KernelFunctions

__global__ void Kernel::Test(DEV_Vector4i* faceNormalIndex, int faceNumber, DEV_Vector3f* worldSpaceNormal)
{
	int faceIndex = blockIdx.x * WARP + threadIdx.x;

	/*printf("%f %f %f\n", worldSpaceNormal[faceNormalIndex[faceIndex].x].x
		, worldSpaceNormal[faceNormalIndex[faceIndex].x].y
		, worldSpaceNormal[faceNormalIndex[faceIndex].x].z);*/
}

__global__ void Kernel::Test2(Depth* depthBuffer, int frameWidth, int frameHeight)
{
	int i = blockIdx.y * WARP + threadIdx.y;
	int j = blockIdx.x * WARP + threadIdx.x;

	if (i >= frameHeight || j >= frameWidth) {
		return;
	}
	int depthIndex = i * frameWidth + j;

	// printf("%f\n", depthBuffer[depthIndex].depth);
}

__global__ void Kernel::WipeFrameBuffer(uchar* frameBuffer, Color skyColor, int frameWidth, int frameHeight)
{
	int i = blockIdx.y * WARP + threadIdx.y;
	int j = blockIdx.x * WARP + threadIdx.x;
	if (i >= frameHeight || j >= frameWidth) {
		return;
	}
	int pixelIndex = i * frameWidth * 3 + j * 3;

	int r = pixelIndex;
	int g = r + 1;
	int b = g + 1;

	frameBuffer[r] = skyColor.r;
	frameBuffer[g] = skyColor.g;
	frameBuffer[b] = skyColor.b;
}

__global__ void Kernel::InitializeDepth(Depth* depthBuffer, int frameWidth, int frameHeight)
{
	int i = blockIdx.y * WARP + threadIdx.y;
	int j = blockIdx.x * WARP + threadIdx.x;
	if (i >= frameHeight || j >= frameWidth) {
		return;
	}
	int pixelIndex = i * frameWidth + j;

	depthBuffer[pixelIndex].depth = 1.0f;
	depthBuffer[pixelIndex].uv.x = -1.0f;
}

__global__ void Kernel::VertexShader(DEV_Vector4f* vertexPosition, DEV_Vector4f* clipSpacePosition, 
	float* MVP, int vertexNumber)
{
	// ��ϰ� ������ �ε����� �� ��° ���ؽ����� ã��
	int vertexIndex = (blockIdx.x * WARP + threadIdx.x) / 4;
	// 0 = x, 1 = y, 2 = z, 3 = w
	int xyzwIndex = threadIdx.x % 4;

	// ���� �� ���� ������ ���� ����
	if (vertexIndex >= vertexNumber) {
		return;
	}
	float sum = 0.0f;

	/*printf("%f %f %f %f\n", 
		vertexPosition[vertexIndex].x, 
		vertexPosition[vertexIndex].y, 
		vertexPosition[vertexIndex].z, 
		vertexPosition[vertexIndex].w);*/

	// MVP * ������ ���
	sum += MVP[xyzwIndex * 4 + 0] * vertexPosition[vertexIndex].x;
	sum += MVP[xyzwIndex * 4 + 1] * vertexPosition[vertexIndex].y;
	sum += MVP[xyzwIndex * 4 + 2] * vertexPosition[vertexIndex].z;
	sum += MVP[xyzwIndex * 4 + 3] * vertexPosition[vertexIndex].w;

	switch (xyzwIndex) {
	case 0:
		clipSpacePosition[vertexIndex].x = sum;
		break;
	case 1:
		clipSpacePosition[vertexIndex].y = sum;
		break;
	case 2:
		clipSpacePosition[vertexIndex].z = sum;
		break;
	case 3:
		clipSpacePosition[vertexIndex].w = sum;
		break;
	default:
		break;
	}
	/*printf("%f %f %f %f\n",
		clipSpacePosition[vertexIndex].x,
		clipSpacePosition[vertexIndex].y,
		clipSpacePosition[vertexIndex].z,
		clipSpacePosition[vertexIndex].w);*/
}

__global__ void Kernel::LocalToWolrdNormal(DEV_Vector3f* vertextNormal, DEV_Vector3f* worldSpaceNormal, float* WSNT, int normalNumber)
{
	int normalIndex = (blockIdx.x * WARP + threadIdx.x) / 3;
	// 0 = x, 1 = y, 2 = z
	int xyzIndex = (blockIdx.x * WARP + threadIdx.x) % 3;

	if (normalIndex >= normalNumber) {
		return;
	}

	// printf("%f %f %f\n", vertextNormal[normalIndex].x, vertextNormal[normalIndex].y, vertextNormal[normalIndex].z);

	float sum = 0.0f;

	sum += WSNT[xyzIndex * 4 + 0] * vertextNormal[normalIndex].x;
	sum += WSNT[xyzIndex * 4 + 1] * vertextNormal[normalIndex].y;
	sum += WSNT[xyzIndex * 4 + 2] * vertextNormal[normalIndex].z;

	switch (xyzIndex) {
	case 0:
		worldSpaceNormal[normalIndex].x = sum;
		break;
	case 1:
		worldSpaceNormal[normalIndex].y = sum;
		break;
	case 2:
		worldSpaceNormal[normalIndex].z = sum;
		break;
	default:
		break;
	}
}

__global__ void Kernel::PerspectiveDivision(DEV_Vector4f* clipSpacePosition, int vertexNumber)
{
	int vertexIndex = (blockIdx.x * WARP) + threadIdx.x;
	if (vertexIndex >= vertexNumber) {
		return;
	}

	clipSpacePosition[vertexIndex].x /= clipSpacePosition[vertexIndex].w;
	clipSpacePosition[vertexIndex].y /= clipSpacePosition[vertexIndex].w;
	clipSpacePosition[vertexIndex].z /= clipSpacePosition[vertexIndex].w;
	clipSpacePosition[vertexIndex].w = 1.0f;

	/*if (-1.0f < clipSpacePosition[vertexIndex].x && clipSpacePosition[vertexIndex].x < 1.0f &&
		- 1.0f < clipSpacePosition[vertexIndex].y && clipSpacePosition[vertexIndex].y < 1.0f &&
		- 1.0f < clipSpacePosition[vertexIndex].z && clipSpacePosition[vertexIndex].z < 1.0f) {
		printf("%.3f %.3f %.3f\n",
			clipSpacePosition[vertexIndex].x,
			clipSpacePosition[vertexIndex].y,
			clipSpacePosition[vertexIndex].z);
	}*/

	/*printf("%f %f %f\n", 
		clipSpacePosition[vertexIndex].x, 
		clipSpacePosition[vertexIndex].y, 
		clipSpacePosition[vertexIndex].z);*/
}

__global__ void Kernel::BackFaceCulling(DEV_Vector4i* facePositionIndex, DEV_Vector4f* clipSpacePosition, 
	bool* shouldRenderFace, int faceNumber)
{
	// �ش� �����尡 ����� face�� Index ���ϱ�
	int faceIndex = (blockIdx.x * WARP) + threadIdx.x;
	if (faceIndex >= faceNumber) {
		return;
	}
	shouldRenderFace[faceIndex] = true;
	// ��Ľ� ������� backface �Ǵ�
	float faceCCW = (clipSpacePosition[facePositionIndex[faceIndex].y].x - clipSpacePosition[facePositionIndex[faceIndex].x].x)
		* (clipSpacePosition[facePositionIndex[faceIndex].z].y - clipSpacePosition[facePositionIndex[faceIndex].x].y)
		- (clipSpacePosition[facePositionIndex[faceIndex].z].x - clipSpacePosition[facePositionIndex[faceIndex].x].x)
		* (clipSpacePosition[facePositionIndex[faceIndex].y].y - clipSpacePosition[facePositionIndex[faceIndex].x].y);
	if (faceCCW < 0) {
		shouldRenderFace[faceIndex] = false;
	}
}

__global__ void Kernel::ViewportTransform(DEV_Vector4f* clipSpacePosition, DEV_Vector4f* viewportSpacePosition,
	float* viewportMatrix, int vertexNumber)
{
	// ��ϰ� ������ �ε����� �� ��° ���ؽ����� ã��
	int vertexIndex = (blockIdx.x * WARP / 4) + (threadIdx.x / 4);
	// 0 = x, 1 = y, 2 = z, 3 = w
	int xyzwIndex = threadIdx.x % 4;

	// ���� �� ���� ������ ���� ����
	if (vertexIndex >= vertexNumber) {
		return;
	}
	float sum = 0.0f;

	/*printf("%f %f %f %f\n",
		clipSpacePosition[vertexIndex].x,
		clipSpacePosition[vertexIndex].y,
		clipSpacePosition[vertexIndex].z,
		clipSpacePosition[vertexIndex].w);*/

	// Viewport ��� * ������ ���
	sum += viewportMatrix[xyzwIndex * 4 + 0] * clipSpacePosition[vertexIndex].x;
	sum += viewportMatrix[xyzwIndex * 4 + 1] * clipSpacePosition[vertexIndex].y;
	sum += viewportMatrix[xyzwIndex * 4 + 2] * clipSpacePosition[vertexIndex].z;
	sum += viewportMatrix[xyzwIndex * 4 + 3] * clipSpacePosition[vertexIndex].w;

	switch (xyzwIndex) {
	case 0:
		viewportSpacePosition[vertexIndex].x = sum;
		break;
	case 1:
		viewportSpacePosition[vertexIndex].y = sum;
		break;
	case 2:
		viewportSpacePosition[vertexIndex].z = sum;
		break;
	case 3:
		viewportSpacePosition[vertexIndex].w = sum;
		break;
	default:
		break;
	}
	/*printf("%f %f %f\n", 
		viewportSpacePosition[vertexIndex].x, 
		viewportSpacePosition[vertexIndex].y, 
		viewportSpacePosition[vertexIndex].z);*/
}

__global__ void Kernel::ScanConversion(DEV_Vector4i* facePositionIndex, DEV_Vector4i* faceUVIndex, DEV_Vector4i* faceNormalIndex,
	DEV_Vector4f* viewportSpacePosition, DEV_Vector3f* vertexUV, DEV_Vector3f* worldSpaceNormal, bool* shouldRenderFace,
	Depth* depthBuffer, int faceNumber, int frameWidth, int frameHeight)
{
	// �ش� �����尡 ����� face�� Index ���ϱ�
	int faceIndex = (blockIdx.x * WARP) + threadIdx.x;

	// ���� �� ��ġ�� ������ ���
	if (faceIndex >= faceNumber) {
		return;
	}
	
	// �����̽��� �ø�
	if (!shouldRenderFace[faceIndex]) {
		return;
	}

	//
	DEV_Vector3f v0 = DEV_Vector3f(viewportSpacePosition[facePositionIndex[faceIndex].x].x,
		viewportSpacePosition[facePositionIndex[faceIndex].x].y,
		viewportSpacePosition[facePositionIndex[faceIndex].x].z);

	DEV_Vector3f v1 = DEV_Vector3f(viewportSpacePosition[facePositionIndex[faceIndex].y].x,
		viewportSpacePosition[facePositionIndex[faceIndex].y].y,
		viewportSpacePosition[facePositionIndex[faceIndex].y].z);

	DEV_Vector3f v2 = DEV_Vector3f(viewportSpacePosition[facePositionIndex[faceIndex].z].x,
		viewportSpacePosition[facePositionIndex[faceIndex].z].y,
		viewportSpacePosition[facePositionIndex[faceIndex].z].z);

	DEV_Vector3f uv0;
	DEV_Vector3f uv1;
	DEV_Vector3f uv2;

	if (faceUVIndex[faceIndex].x == -1) {
		uv0.x = -1.0f;
		uv1.x = -1.0f;
		uv2.x = -1.0f;
	}
	else {

		uv0 = vertexUV[faceUVIndex[faceIndex].x];
		uv1 = vertexUV[faceUVIndex[faceIndex].y];
		uv2 = vertexUV[faceUVIndex[faceIndex].z];
	}

	// y��ǥ ���� ���� ���� �ϵ� �ڵ�
	if (v0.y > v1.y) {
		Swap(&v0, &v1);
		Swap(&uv0, &uv1);
	}
	if (v1.y > v2.y) {
		Swap(&v1, &v2);
		Swap(&uv1, &uv2);
	}
	if (v0.y > v1.y) {
		Swap(&v0, &v1);
		Swap(&uv0, &uv1);
	}

	// �� ���ؽ����� ���ؽ������� ����
	DEV_Vector3f vector0to1 = v1 - v0;
	DEV_Vector3f vector0to2 = v2 - v0;
	DEV_Vector3f vector1to2 = v2 - v1;

	// �̸� ���� ����
	float rsqrt0to1 = FastReverseSqrt(vector0to1.x * vector0to1.x + 
		vector0to1.y * vector0to1.y +
		vector0to1.z * vector0to1.z);
	float rsqrt0to2 = FastReverseSqrt(vector0to2.x * vector0to2.x +
		vector0to2.y * vector0to2.y +
		vector0to2.z * vector0to2.z);
	float rsqrt1to2 = FastReverseSqrt(vector1to2.x * vector1to2.x +
		vector1to2.y * vector1to2.y +
		vector1to2.z * vector1to2.z);

	// y Ŭ����
	int y0 = Clamp(v0.y, 0.0f, frameHeight -1.0f);
	int y1 = Clamp(v1.y, 0.0f, frameHeight - 1.0f);
	int y2 = Clamp(v2.y, 0.0f, frameHeight - 1.0f);

	if (y0 >= frameHeight - 1) return; // ���� ���� ���� ȭ�� ���̸� ����

	// y0 -> y1 ���� ��ĳ��
	int yScanCount = 0;
	// ȭ�� �Ʒ����� ������ ��� ��ĵ ī��Ʈ �ø��� ����
	if (v0.y < 0.0f) {
		yScanCount = y0 - (int)v0.y;
	}

	for (int pixelY = y0; pixelY < y1; ++pixelY, ++yScanCount) {
		// printf("%d : ", pixelY);
		
		// y�� ���̰� �̼��� ��� �����׸�Ʈ �������� ����
		if (vector0to1.y < 0.5f || vector0to2.y < 0.5f) {
			break;
		}

		// ��ĵ ���� �� �� ���� ã��
		DEV_Vector3f scan0 = v0 + ((vector0to1 / vector0to1.y) * yScanCount);
		DEV_Vector3f scan1 = v0 + ((vector0to2 / vector0to2.y) * yScanCount);

		float uvLerpDegree0to1 = (scan0 - v0).GetLength() * rsqrt0to1;
		float uvLerpDegree0to2 = (scan1 - v0).GetLength() * rsqrt0to2;

		// ��ĵ ���� UV ����
		DEV_Vector3f scan0LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree0to1, uv0.x, uv1.x),
			Lerp(uvLerpDegree0to1, uv0.y, uv1.y), 
			-1);

		DEV_Vector3f scan1LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree0to2, uv0.x, uv2.x),
			Lerp(uvLerpDegree0to2, uv0.y, uv2.y), 
			-1);

		// xũ��� ���� (���ʿ��� ������ ��ĵ)
		if (scan0.x > scan1.x) {
			Swap(&scan0, &scan1);
			Swap(&scan0LerpedUV, &scan1LerpedUV);
		}
		// ��ĵ ���������� xz��� ����
		DEV_Vector3f scanVec = scan1 - scan0;

		// x Ŭ����
		int start = Clamp(scan0.x, 0.0f, frameWidth - 1.0f);
		int end = Clamp(scan1.x, 0.0f, frameWidth - 1.0f);

		int j = 0;
		if (scan0.x < 0.0f) {
			j = start - scan0.x;
		}

		// zŬ���� �ϸ鼭 �����׸�Ʈ ä���
		for (int pixelX = start; pixelX < end; ++pixelX, ++j) {


			// x�� ���̰� �̼��� ��� �����׸�Ʈ �������� ����
			if (scanVec.x < 0.5f) {
				break;
			}

			DEV_Vector3f scanX = scan0 + (scanVec / scanVec.x) * j;
			float z = scanX.z;

			if (0.0f < z && z <= 1.0f) {
				// printf("x : %d, z : %f ", pixelX, z);
				// �����׸�Ʈ UV ����
				DEV_Vector3f lerpedUV = DEV_Vector3f(Lerp((scanX.x - scan0.x) / scanVec.x, scan0LerpedUV.x, scan1LerpedUV.x),
					Lerp((scanX.x - scan0.x) / scanVec.x, scan0LerpedUV.y, scan1LerpedUV.y),
					-1);


				WriteDepth(depthBuffer, z, ((frameHeight - pixelY - 1)* frameWidth) + pixelX,
					frameWidth * frameHeight, faceIndex, faceNormalIndex, 
					lerpedUV, worldSpaceNormal);
			}
		}
		// printf("\n");
	}
	// y1 -> y2 ���� ��ĳ��
	int i = 0;
	if (v1.y < 0.0f) {
		i = y1 - v1.y;
	}

	for (int pixelY = y1; pixelY < y2; ++pixelY, ++i, ++yScanCount) {
		// printf("%d : ", pixelY);
		// printf("%d %d\n", pixelY, y1);

		// y�� ���̰� �̼��� ��� �����׸�Ʈ �������� ����
		if (vector1to2.y < 0.5f || vector0to2.y < 0.5f) {
			break;
		}

		// ��ĵ ���� ã��
		DEV_Vector3f scan0 = v0 + ((vector0to2 / vector0to2.y) * yScanCount); // 0 ������ ����
		DEV_Vector3f scan1 = v1 + ((vector1to2 / vector1to2.y) * i); // 0 ������ ����

		float uvLerpDegree0to2 = (scan0 - v0).GetLength() * rsqrt0to2;
		float uvLerpDegree1to2 = (scan1 - v1).GetLength() * rsqrt1to2;

		// ��ĵ ���� UV ����
		DEV_Vector3f scan0LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree0to2, uv0.x, uv2.x),
			Lerp(uvLerpDegree0to2, uv0.y, uv2.y),
			-1);

		DEV_Vector3f scan1LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree1to2, uv1.x, uv2.x),
			Lerp(uvLerpDegree1to2, uv1.y, uv2.y),
			-1);

		// xũ��� ���� (���ʿ��� ������ ��ĵ)
		if (scan0.x > scan1.x) {
			Swap(&scan0, &scan1);
			Swap(&scan0LerpedUV, &scan1LerpedUV);
		}
		// ��ĵ ���������� xz��� ����
		DEV_Vector3f scanVec = scan1 - scan0;

		// x Ŭ����
		int start = Clamp(scan0.x, 0.0f, frameWidth - 1.0f);
		int end = Clamp(scan1.x, 0.0f, frameWidth - 1.0f);

		int j = 0;
		if (scan0.x < 0.0f) {
			j = start - (int)(scan0.x + 0.5f);
		}

		// zŬ���� �ϸ鼭 �����׸�Ʈ ä���
		for (int pixelX = start; pixelX < end; ++pixelX, ++j) {

			// x�� ���̰� �̼��� ��� �����׸�Ʈ �������� ����
			if (scanVec.x < 0.5f) {
				break;
			}

			DEV_Vector3f scanX = scan0 + (scanVec / scanVec.x) * j;
			float z = scanX.z;

			if (0.0f < z && z <= 1.0f) {
				// printf("x : %d, z : %d ", pixelX, z);
				// �����׸�Ʈ UV ����
				DEV_Vector3f lerpedUV = DEV_Vector3f(Lerp((scanX.x - scan0.x) / (scan1.x - scan0.x), scan0LerpedUV.x, scan1LerpedUV.x),
					Lerp((scanX.x - scan0.x) / (scan1.x - scan0.x), scan0LerpedUV.y, scan1LerpedUV.y),
					-1);

				WriteDepth(depthBuffer, z, ((frameHeight - pixelY - 1) * frameWidth) + pixelX,
					frameWidth * frameHeight, faceIndex, faceNormalIndex,
					lerpedUV, worldSpaceNormal);
			}
		}
		// printf("\n");
	}
}

__global__ void Kernel::FragmentShader(Depth* depthBuffer, uchar* frameBuffer, uchar* texture, 
	DirectionalLight mainLight, Color skyColor, int frameWidth, int frameHeight, int textureWidth, int textureHeight)
{
	int i = blockIdx.y * WARP + threadIdx.y;
	int j = blockIdx.x * WARP + threadIdx.x;

	if (i >= frameHeight || j >= frameWidth) {
		return;
	}
	int depthIndex = i * frameWidth + j;

	// depth�� 1���� ���� �ȼ��� ��ο�
	if (depthBuffer[depthIndex].depth > 0.999f) {
		// printf(" no draw ");
		return;
	}

	int pixelIndex = i * frameWidth * 3 + j * 3;

	int r = pixelIndex;
	int g = r + 1;
	int b = g + 1;

	if (depthBuffer[depthIndex].uv.x < 0.0f) {
		// UV ������ ���� �÷� �Ҵ�
		frameBuffer[r] = 0;
		frameBuffer[g] = 250;
		frameBuffer[b] = 0;

		// printf(" no UV ");

		return;
	};

	/*printf(" (%f %f %f) ", 
		depthBuffer[depthIndex].normal.x, 
		depthBuffer[depthIndex].normal.y, 
		depthBuffer[depthIndex].normal.z);*/

	// ���� �븻 0���� �����ϱ� ��ġ�� ����
	float NDotL = Clamp((depthBuffer[depthIndex].normal * mainLight.direction), 0.0f, 1.0f);

	int uvX = Clamp(depthBuffer[depthIndex].uv.x * textureWidth, 0.0f, textureWidth - 1.0f);
	int uvY = Clamp((1.0f - depthBuffer[depthIndex].uv.y) * textureHeight, 0.0f, textureHeight - 1.0f);

	float lightR = mainLight.rgb[0] / 255.0f;
	float lightG = mainLight.rgb[1] / 255.0f;
	float lightB = mainLight.rgb[2] / 255.0f;
	// printf(" (%f %f %f) ", lightR, lightG, lightB);

	// �ֺ���(ambient)�� ���߿�

	// ���ݻ�(diffuse)
	// �ؽ��Ĵ� BGR�� ���� (cv::Mat)
	// �븻 ��� ���� �߰� X
	uchar diffuseR = texture[uvY * textureWidth * 3 + uvX * 3 + 2] * lightR;// * NDotL;
	uchar diffuseG = texture[uvY * textureWidth * 3 + uvX * 3 + 1] * lightG;// * NDotL;
	uchar diffuseB = texture[uvY * textureWidth * 3 + uvX * 3] * lightB;// * NDotL;

	//// ���ݻ�(specular)
	//// ��ĵ ���������� Position ���� �߰��ؼ� View ���� ����� �ϱ�

	frameBuffer[r] = diffuseR;
	frameBuffer[g] = diffuseG;
	frameBuffer[b] = diffuseB;
}


#pragma endregion
#pragma region GPUMemoryMangement
void Kernel::MAllocOneFrameElementsToDevice(int frameSize)
{
	// ��� ������ ���� �޸� �Ҵ�
	cudaMalloc((void**)&dev_frameBuffer, sizeof(unsigned char) * frameSize * 3);
	cudaMalloc((void**)&dev_depthBuffer, sizeof(Depth) * frameSize);

	// MVP ��� ���� �޸� �Ҵ�
	cudaMalloc((void**)&dev_MVP, sizeof(float) * 4 * 4);
	// ����Ʈ ���
	cudaMalloc((void**)&dev_viewportMatrix, sizeof(float) * 4 * 4);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "SeOneFrameElements cudaMalloc fail..." << cudaStatus << "\n";
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}
}

void Kernel::SetViewportMatrixToDevice(cv::Mat& viewportMatrix)
{
	// GPU�� ���縦 ���ؼ� �޸� ���Ӽ� �˻�
	if (!viewportMatrix.isContinuous()) {
		viewportMatrix = viewportMatrix.clone();
	}

	cudaMemcpy(dev_viewportMatrix, reinterpret_cast<float*>(viewportMatrix.data), sizeof(float) * 4 * 4, cudaMemcpyHostToDevice);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "SetViewportMatrix cudaMemcpy fail..." << cudaStatus << "\n";
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}
}

void Kernel::SetMVPToDevice(cv::Mat& MVP)
{
	// GPU�� ���縦 ���ؼ� �޸� ���Ӽ� �˻�
	if (!MVP.isContinuous()) {
		MVP = MVP.clone();
	}

	cudaMemcpy(dev_MVP, reinterpret_cast<float*>(MVP.data), sizeof(float) * 4 * 4, cudaMemcpyHostToDevice);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "SetMVp cudaMemcpy fail..." << cudaStatus << "\n";
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}
}

void Kernel::SetWorldSpaceNormalTransform(cv::Mat& WSNT)
{
	cudaMalloc((void**)&dev_worldSpaceNormalTransform, sizeof(float) * 4 * 4);
	// GPU�� ���縦 ���ؼ� �޸� ���Ӽ� �˻�
	if (!WSNT.isContinuous()) {
		WSNT = WSNT.clone();
	}

	cudaMemcpy(dev_worldSpaceNormalTransform, 
		reinterpret_cast<float*>(WSNT.data),
		sizeof(float) * 4 * 4, cudaMemcpyHostToDevice);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "SetWorldSpaceNormalTransform fail..." << cudaStatus << "\n";
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}
}

void Kernel::SetModelDataToDevice(const std::shared_ptr<Model>& model)
{
	// ���ؽ� ���� ���� �޸� �Ҵ�
	cudaMalloc((void**)&dev_vertexPosition, sizeof(DEV_Vector4f) * model->vertexNumber);
	cudaMalloc((void**)&dev_vertexUV, sizeof(DEV_Vector3f) * model->uvNumber);
	cudaMalloc((void**)&dev_vertexNormal, sizeof(DEV_Vector3f) * model->vertexNormalNumber);

	// ���̽� ���� ���� �޸� �Ҵ�
	cudaMalloc((void**)&dev_facePositionIndex, sizeof(DEV_Vector4i) * model->faceNumber);
	cudaMalloc((void**)&dev_faceUVIndex, sizeof(DEV_Vector4i) * model->faceNumber);
	cudaMalloc((void**)&dev_faceNormalIndex, sizeof(DEV_Vector4i) * model->faceNumber);

	// �ؽ��� �޸� �Ҵ�
	if (!model->texture.isContinuous()) { // �����ϱ� ���� �޸� ���Ӽ� ����
		model->texture = model->texture.clone();
	}
	cudaMalloc((void**)&dev_texture, sizeof(uchar) * 3 * model->texture.total());

	// ��� ������ ���� �޸� �Ҵ�
	cudaMalloc((void**)&dev_worldSpaceNormal, sizeof(DEV_Vector3f) * model->vertexNormalNumber);
	cudaMalloc((void**)&dev_clipSpacePosition, sizeof(DEV_Vector4f) * model->vertexNumber);
	cudaMalloc((void**)&dev_viewportSpacePosition, sizeof(DEV_Vector4f) * model->vertexNumber);
	cudaMalloc((void**)&dev_shouldRenderFace, sizeof(bool) * model->faceNumber);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Model data cudaMalloc fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// ������ CPU->GPU�� ����
	// ���� ���ؽ� ������ ����
	cudaMemcpy(dev_vertexPosition,
		model->vertexPosition.data(),
		sizeof(DEV_Vector4f) * model->vertexNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertexUV,
		model->vertexUV.data(),
		sizeof(DEV_Vector3f) * model->uvNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertexNormal,
		model->vertexNormal.data(),
		sizeof(DEV_Vector3f) * model->vertexNormalNumber, cudaMemcpyHostToDevice);


	// ���̽�(������) ������ ����
	cudaMemcpy(dev_facePositionIndex,
		model->facePositionIndex.data(),
		sizeof(DEV_Vector4i) * model->faceNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_faceUVIndex,
		model->faceUVIndex.data(),
		sizeof(DEV_Vector4i) * model->faceNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_faceNormalIndex,
		model->faceNormalIndex.data(),
		sizeof(DEV_Vector4i) * model->faceNumber, cudaMemcpyHostToDevice);
	// �ؽ��� ������ ����
	cudaMemcpy(dev_texture, model->texture.data,
		sizeof(uchar) * 3 * model->texture.total(), cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Model data cudaMemcpy fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	
}

void Kernel::DeleteModelMemory()
{
	cudaFree(dev_vertexPosition);
	cudaFree(dev_vertexUV);
	cudaFree(dev_vertexNormal);

	cudaFree(dev_facePositionIndex);
	cudaFree(dev_faceUVIndex);
	cudaFree(dev_faceNormalIndex);
	cudaFree(dev_texture);

	cudaFree(dev_worldSpaceNormal);
	cudaFree(dev_clipSpacePosition);
	cudaFree(dev_viewportSpacePosition);
	cudaFree(dev_shouldRenderFace);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "1DeleteModelMemory cudaFree fail..." << cudaStatus << "\n";

		exit(EXIT_FAILURE);
	}
}

void Kernel::DeleteOneFrameElements()
{
	cudaFree(dev_frameBuffer);
	cudaFree(dev_depthBuffer);

	cudaFree(dev_MVP);
	cudaFree(dev_viewportMatrix);
	cudaFree(dev_worldSpaceNormalTransform);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "DeleteOneFrameElements cudaFree fail..." << cudaStatus << "\n";
		exit(EXIT_FAILURE);
	}
}
void Kernel::CopyResultToMonitor(uchar* camBufferPtr, int frameBufferSize)
{
	// GPU �޸� -> CPU �޸�
	cudaError_t cudaStatus = cudaMemcpy(camBufferPtr, dev_frameBuffer, sizeof(uchar) * frameBufferSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("GPU to CPU cudaMemcpy fail...\n");
		exit(EXIT_FAILURE);
	}
}
#pragma endregion
#pragma region WrapperFunctions
void Kernel::InitializeFrame(Color skyColor, int frameWidth, int frameHeight)
{
	int gridX = frameWidth % WARP == 0 ? frameWidth / WARP : frameWidth / WARP + 1;
	int gridY = frameHeight % WARP == 0 ? frameHeight / WARP : frameHeight / WARP + 1;

	dim3 grid(gridX, gridY);
	dim3 block(WARP, WARP);
	WipeFrameBuffer<<<grid, block>>>(dev_frameBuffer, skyColor, frameWidth, frameHeight);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "WipeFrame cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	InitializeDepth << <grid, block >> > (dev_depthBuffer, frameWidth, frameHeight);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "InitializeDepth cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// std::cout << "������ ���� �ʱ�ȭ...\n";
}

void Kernel::StartVertexShader(int vertexNumber, int vertexNormalNumber)
{
	// ������ ��� ������
	// ���ؽ� �ϳ��� ������ 4��, WARP 32�϶� ��� �ϳ��� ���ؽ� 8��
	int gridX = ((vertexNumber * 4) % WARP) == 0 ? vertexNumber * 4 / WARP : (vertexNumber * 4 / WARP) + 1;
	dim3 grid(gridX, 1);
	dim3 block(WARP, 1); // ���������� ¥����

	// MVP ��ȯ ����
	VertexShader<<<grid, block >>>(dev_vertexPosition, dev_clipSpacePosition, dev_MVP, vertexNumber);
	// ����ȭ
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "VertexShader cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	gridX = ((vertexNumber * 3) % WARP) == 0 ? vertexNumber * 3 / WARP : (vertexNumber * 3 / WARP) + 1;
	grid.x = gridX;
	
	// ����Ʈ ����� ���� �븻 ���彺���̽� �������� �ٲٱ�
	LocalToWolrdNormal<<<grid, block>>>(dev_vertexNormal, dev_worldSpaceNormal, 
		dev_worldSpaceNormalTransform, vertexNormalNumber);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "LocalToWolrdNormal cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// std::cout << "���ؽ� ���̴������� �� ���� �Ǿ���...\n";
}

void Kernel::StartRasterizer(int vertexNumber, int faceNumber, int frameWidth, int frameHeight)
{
	// ������ ��� ������
	// ���ؽ� �ϳ��� ������ 1��, WARP 32�϶� ��� �ϳ��� ���ؽ� 32��
	int gridX = vertexNumber % WARP == 0 ? vertexNumber / WARP : (vertexNumber / WARP) + 1;
	dim3 grid(gridX, 1);
	dim3 block(WARP, 1); // ���������� ¥����

	// ���� ������ ����
	PerspectiveDivision<<<grid, block >>>(dev_clipSpacePosition, vertexNumber);
	// ����ȭ
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "PerspectiveDivision cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// Ŭ���� �� �ϰ� ��ĵ ���������� �ڸ��� ������� �ϱ�
	// ���̽� �ϳ��� ������ �Ѱ� WARP 32�϶� ��� �ϳ��� ���̽� 32��
	gridX = faceNumber % WARP == 0 ? faceNumber / WARP : (faceNumber / WARP) + 1;
	grid.x = gridX;
	// �����̽� �ø�

	BackFaceCulling<<<grid, block >>>(dev_facePositionIndex, dev_clipSpacePosition, dev_shouldRenderFace, faceNumber);

	// ����ȭ
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	    std::cout << "BackFaceCulling cudaDeviceSynchronize fail..." << cudaStatus << "\n";
	    DeleteModelMemory();
	    DeleteOneFrameElements();
	    exit(EXIT_FAILURE);
	}

	// ���ؽ� �ϳ��� ������ 4��, WARP 32�϶� ��� �ϳ��� ���ؽ� 8��
	gridX = ((vertexNumber * 4) % WARP) == 0 ? vertexNumber * 4 / WARP : (vertexNumber * 4 / WARP) + 1;
	grid.x = gridX;
	// ����Ʈ ��ȯ
	ViewportTransform<<<grid, block>>>(dev_clipSpacePosition, dev_viewportSpacePosition, dev_viewportMatrix, vertexNumber);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "ViewportTransform cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// ��ĵ ������
	// ���̽� �ϳ��� ������ �Ѱ� WARP 32�϶� ��� �ϳ��� ���̽� 32��
	gridX = faceNumber % WARP == 0 ? faceNumber / WARP : (faceNumber / WARP) + 1;
	grid.x = gridX;

	ScanConversion<<<grid, block>>>(dev_facePositionIndex, dev_faceUVIndex, dev_faceNormalIndex,
		dev_viewportSpacePosition, dev_vertexUV, dev_worldSpaceNormal, dev_shouldRenderFace, 
		dev_depthBuffer, faceNumber, frameWidth, frameHeight);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "ScanConversion cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// std::cout << "�����Ͷ����� ���� ���� \n";
}

void Kernel::StartFragmentShader(DirectionalLight mainLight, Color skyColor, 
	int frameWidth, int frameHeight, int textureWidth, int textureHeight)
{
	int gridX = frameWidth % WARP == 0 ? frameWidth / WARP : frameWidth / WARP + 1;
	int gridY = frameHeight % WARP == 0 ? frameHeight / WARP : frameHeight / WARP + 1;

	dim3 grid(gridX, gridY);
	dim3 block(WARP, WARP);

	FragmentShader<<<grid, block>>>(dev_depthBuffer, dev_frameBuffer, dev_texture, 
		mainLight, skyColor, frameWidth, frameHeight, textureWidth, textureHeight);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "FragmentShader cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Rendering fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}
	
	// std::cout << "������Ʈ �ϳ� �� �׷ȴ�...\n";
}
#pragma endregion
