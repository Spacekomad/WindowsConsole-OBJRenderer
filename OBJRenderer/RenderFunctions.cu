#include "RenderFunctions.h"

using namespace Kernel;

#pragma region DeviceVariables
__device__ DEV_Vector4f* dev_vertexPosition;
__device__ DEV_Vector3f* dev_vertexUV;
__device__ DEV_Vector3f* dev_vertexNormal;

// 메모리 복사를 1열로 하는 게 제일 빠르기 때문에 나눠야 됨
__device__ DEV_Vector4i* dev_facePositionIndex;
__device__ DEV_Vector4i* dev_faceUVIndex;
__device__ DEV_Vector4i* dev_faceNormalIndex;
__device__ uchar* dev_texture;

__device__ float* dev_MVP;
__device__ float* dev_viewportMatrix;
__device__ float* dev_worldSpaceNormalTransform;

// 결과를 담을 디바이스 변수들
__device__ bool* dev_shouldRenderFace;
__device__ DEV_Vector3f* dev_worldSpaceNormal;
__device__ DEV_Vector4f* dev_clipSpacePosition;
__device__ DEV_Vector4f* dev_viewportSpacePosition;
__device__ Depth* dev_depthBuffer;
// frameBuffer 구조 => 0번 픽셀 r, g, b -> 1번 픽셀 r, g, b... 따라서 0~2까지 1번 픽셀
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

// 고속 역 제곱근
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
	// 블록과 스레드 인덱스로 몇 번째 버텍스인지 찾기
	int vertexIndex = (blockIdx.x * WARP + threadIdx.x) / 4;
	// 0 = x, 1 = y, 2 = z, 3 = w
	int xyzwIndex = threadIdx.x % 4;

	// 와프 내 남는 스레드 먼저 종료
	if (vertexIndex >= vertexNumber) {
		return;
	}
	float sum = 0.0f;

	/*printf("%f %f %f %f\n", 
		vertexPosition[vertexIndex].x, 
		vertexPosition[vertexIndex].y, 
		vertexPosition[vertexIndex].z, 
		vertexPosition[vertexIndex].w);*/

	// MVP * 포지션 행렬
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
	// 해당 스레드가 담당한 face의 Index 구하기
	int faceIndex = (blockIdx.x * WARP) + threadIdx.x;
	if (faceIndex >= faceNumber) {
		return;
	}
	shouldRenderFace[faceIndex] = true;
	// 행렬식 계산으로 backface 판단
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
	// 블록과 스레드 인덱스로 몇 번째 버텍스인지 찾기
	int vertexIndex = (blockIdx.x * WARP / 4) + (threadIdx.x / 4);
	// 0 = x, 1 = y, 2 = z, 3 = w
	int xyzwIndex = threadIdx.x % 4;

	// 와프 내 남는 스레드 먼저 종료
	if (vertexIndex >= vertexNumber) {
		return;
	}
	float sum = 0.0f;

	/*printf("%f %f %f %f\n",
		clipSpacePosition[vertexIndex].x,
		clipSpacePosition[vertexIndex].y,
		clipSpacePosition[vertexIndex].z,
		clipSpacePosition[vertexIndex].w);*/

	// Viewport 행렬 * 포지션 행렬
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
	// 해당 스레드가 담당한 face의 Index 구하기
	int faceIndex = (blockIdx.x * WARP) + threadIdx.x;

	// 와프 내 넘치는 스레드 대기
	if (faceIndex >= faceNumber) {
		return;
	}
	
	// 백페이스는 컬링
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

	// y좌표 기준 버블 정렬 하드 코딩
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

	// 각 버텍스에서 버텍스까지의 벡터
	DEV_Vector3f vector0to1 = v1 - v0;
	DEV_Vector3f vector0to2 = v2 - v0;
	DEV_Vector3f vector1to2 = v2 - v1;

	// 미리 계산된 역수
	float rsqrt0to1 = FastReverseSqrt(vector0to1.x * vector0to1.x + 
		vector0to1.y * vector0to1.y +
		vector0to1.z * vector0to1.z);
	float rsqrt0to2 = FastReverseSqrt(vector0to2.x * vector0to2.x +
		vector0to2.y * vector0to2.y +
		vector0to2.z * vector0to2.z);
	float rsqrt1to2 = FastReverseSqrt(vector1to2.x * vector1to2.x +
		vector1to2.y * vector1to2.y +
		vector1to2.z * vector1to2.z);

	// y 클리핑
	int y0 = Clamp(v0.y, 0.0f, frameHeight -1.0f);
	int y1 = Clamp(v1.y, 0.0f, frameHeight - 1.0f);
	int y2 = Clamp(v2.y, 0.0f, frameHeight - 1.0f);

	if (y0 >= frameHeight - 1) return; // 제일 작은 값이 화면 밖이면 종료

	// y0 -> y1 까지 스캐닝
	int yScanCount = 0;
	// 화면 아래에서 시작한 경우 스캔 카운트 올리고 시작
	if (v0.y < 0.0f) {
		yScanCount = y0 - (int)v0.y;
	}

	for (int pixelY = y0; pixelY < y1; ++pixelY, ++yScanCount) {
		// printf("%d : ", pixelY);
		
		// y값 차이가 미세한 경우 프래그먼트 생성하지 않음
		if (vector0to1.y < 0.5f || vector0to2.y < 0.5f) {
			break;
		}

		// 스캔 시작 및 끝 지점 찾기
		DEV_Vector3f scan0 = v0 + ((vector0to1 / vector0to1.y) * yScanCount);
		DEV_Vector3f scan1 = v0 + ((vector0to2 / vector0to2.y) * yScanCount);

		float uvLerpDegree0to1 = (scan0 - v0).GetLength() * rsqrt0to1;
		float uvLerpDegree0to2 = (scan1 - v0).GetLength() * rsqrt0to2;

		// 스캔 지점 UV 보간
		DEV_Vector3f scan0LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree0to1, uv0.x, uv1.x),
			Lerp(uvLerpDegree0to1, uv0.y, uv1.y), 
			-1);

		DEV_Vector3f scan1LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree0to2, uv0.x, uv2.x),
			Lerp(uvLerpDegree0to2, uv0.y, uv2.y), 
			-1);

		// x크기로 정렬 (왼쪽에서 오른쪽 스캔)
		if (scan0.x > scan1.x) {
			Swap(&scan0, &scan1);
			Swap(&scan0LerpedUV, &scan1LerpedUV);
		}
		// 스캔 지점끼리의 xz평면 벡터
		DEV_Vector3f scanVec = scan1 - scan0;

		// x 클리핑
		int start = Clamp(scan0.x, 0.0f, frameWidth - 1.0f);
		int end = Clamp(scan1.x, 0.0f, frameWidth - 1.0f);

		int j = 0;
		if (scan0.x < 0.0f) {
			j = start - scan0.x;
		}

		// z클리핑 하면서 프래그먼트 채우기
		for (int pixelX = start; pixelX < end; ++pixelX, ++j) {


			// x값 차이가 미세한 경우 프래그먼트 생성하지 않음
			if (scanVec.x < 0.5f) {
				break;
			}

			DEV_Vector3f scanX = scan0 + (scanVec / scanVec.x) * j;
			float z = scanX.z;

			if (0.0f < z && z <= 1.0f) {
				// printf("x : %d, z : %f ", pixelX, z);
				// 프래그먼트 UV 보간
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
	// y1 -> y2 까지 스캐닝
	int i = 0;
	if (v1.y < 0.0f) {
		i = y1 - v1.y;
	}

	for (int pixelY = y1; pixelY < y2; ++pixelY, ++i, ++yScanCount) {
		// printf("%d : ", pixelY);
		// printf("%d %d\n", pixelY, y1);

		// y값 차이가 미세한 경우 프래그먼트 생성하지 않음
		if (vector1to2.y < 0.5f || vector0to2.y < 0.5f) {
			break;
		}

		// 스캔 지점 찾기
		DEV_Vector3f scan0 = v0 + ((vector0to2 / vector0to2.y) * yScanCount); // 0 나누기 없음
		DEV_Vector3f scan1 = v1 + ((vector1to2 / vector1to2.y) * i); // 0 나누기 없음

		float uvLerpDegree0to2 = (scan0 - v0).GetLength() * rsqrt0to2;
		float uvLerpDegree1to2 = (scan1 - v1).GetLength() * rsqrt1to2;

		// 스캔 지점 UV 보간
		DEV_Vector3f scan0LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree0to2, uv0.x, uv2.x),
			Lerp(uvLerpDegree0to2, uv0.y, uv2.y),
			-1);

		DEV_Vector3f scan1LerpedUV = DEV_Vector3f(Lerp(uvLerpDegree1to2, uv1.x, uv2.x),
			Lerp(uvLerpDegree1to2, uv1.y, uv2.y),
			-1);

		// x크기로 정렬 (왼쪽에서 오른쪽 스캔)
		if (scan0.x > scan1.x) {
			Swap(&scan0, &scan1);
			Swap(&scan0LerpedUV, &scan1LerpedUV);
		}
		// 스캔 지점끼리의 xz평면 벡터
		DEV_Vector3f scanVec = scan1 - scan0;

		// x 클리핑
		int start = Clamp(scan0.x, 0.0f, frameWidth - 1.0f);
		int end = Clamp(scan1.x, 0.0f, frameWidth - 1.0f);

		int j = 0;
		if (scan0.x < 0.0f) {
			j = start - (int)(scan0.x + 0.5f);
		}

		// z클리핑 하면서 프래그먼트 채우기
		for (int pixelX = start; pixelX < end; ++pixelX, ++j) {

			// x값 차이가 미세한 경우 프래그먼트 생성하지 않음
			if (scanVec.x < 0.5f) {
				break;
			}

			DEV_Vector3f scanX = scan0 + (scanVec / scanVec.x) * j;
			float z = scanX.z;

			if (0.0f < z && z <= 1.0f) {
				// printf("x : %d, z : %d ", pixelX, z);
				// 프래그먼트 UV 보간
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

	// depth가 1보다 작은 픽셀만 드로우
	if (depthBuffer[depthIndex].depth > 0.999f) {
		// printf(" no draw ");
		return;
	}

	int pixelIndex = i * frameWidth * 3 + j * 3;

	int r = pixelIndex;
	int g = r + 1;
	int b = g + 1;

	if (depthBuffer[depthIndex].uv.x < 0.0f) {
		// UV 없으면 에러 컬러 할당
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

	// 월드 노말 0으로 나오니까 고치고 쓰기
	float NDotL = Clamp((depthBuffer[depthIndex].normal * mainLight.direction), 0.0f, 1.0f);

	int uvX = Clamp(depthBuffer[depthIndex].uv.x * textureWidth, 0.0f, textureWidth - 1.0f);
	int uvY = Clamp((1.0f - depthBuffer[depthIndex].uv.y) * textureHeight, 0.0f, textureHeight - 1.0f);

	float lightR = mainLight.rgb[0] / 255.0f;
	float lightG = mainLight.rgb[1] / 255.0f;
	float lightB = mainLight.rgb[2] / 255.0f;
	// printf(" (%f %f %f) ", lightR, lightG, lightB);

	// 주변광(ambient)은 나중에

	// 난반사(diffuse)
	// 텍스쳐는 BGR로 받음 (cv::Mat)
	// 노말 계산 아직 추가 X
	uchar diffuseR = texture[uvY * textureWidth * 3 + uvX * 3 + 2] * lightR;// * NDotL;
	uchar diffuseG = texture[uvY * textureWidth * 3 + uvX * 3 + 1] * lightG;// * NDotL;
	uchar diffuseB = texture[uvY * textureWidth * 3 + uvX * 3] * lightB;// * NDotL;

	//// 정반사(specular)
	//// 스캔 컨버전에서 Position 보간 추가해서 View 벡터 만들면 하기

	frameBuffer[r] = diffuseR;
	frameBuffer[g] = diffuseG;
	frameBuffer[b] = diffuseB;
}


#pragma endregion
#pragma region GPUMemoryMangement
void Kernel::MAllocOneFrameElementsToDevice(int frameSize)
{
	// 결과 프레임 버퍼 메모리 할당
	cudaMalloc((void**)&dev_frameBuffer, sizeof(unsigned char) * frameSize * 3);
	cudaMalloc((void**)&dev_depthBuffer, sizeof(Depth) * frameSize);

	// MVP 행렬 담을 메모리 할당
	cudaMalloc((void**)&dev_MVP, sizeof(float) * 4 * 4);
	// 뷰포트 행렬
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
	// GPU로 복사를 위해서 메모리 연속성 검사
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
	// GPU로 복사를 위해서 메모리 연속성 검사
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
	// GPU로 복사를 위해서 메모리 연속성 검사
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
	// 버텍스 정보 담을 메모리 할당
	cudaMalloc((void**)&dev_vertexPosition, sizeof(DEV_Vector4f) * model->vertexNumber);
	cudaMalloc((void**)&dev_vertexUV, sizeof(DEV_Vector3f) * model->uvNumber);
	cudaMalloc((void**)&dev_vertexNormal, sizeof(DEV_Vector3f) * model->vertexNormalNumber);

	// 페이스 정보 담을 메모리 할당
	cudaMalloc((void**)&dev_facePositionIndex, sizeof(DEV_Vector4i) * model->faceNumber);
	cudaMalloc((void**)&dev_faceUVIndex, sizeof(DEV_Vector4i) * model->faceNumber);
	cudaMalloc((void**)&dev_faceNormalIndex, sizeof(DEV_Vector4i) * model->faceNumber);

	// 텍스쳐 메모리 할당
	if (!model->texture.isContinuous()) { // 복사하기 전에 메모리 연속성 검증
		model->texture = model->texture.clone();
	}
	cudaMalloc((void**)&dev_texture, sizeof(uchar) * 3 * model->texture.total());

	// 결과 정보를 담을 메모리 할당
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

	// 데이터 CPU->GPU로 복사
	// 실제 버텍스 데이터 전달
	cudaMemcpy(dev_vertexPosition,
		model->vertexPosition.data(),
		sizeof(DEV_Vector4f) * model->vertexNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertexUV,
		model->vertexUV.data(),
		sizeof(DEV_Vector3f) * model->uvNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertexNormal,
		model->vertexNormal.data(),
		sizeof(DEV_Vector3f) * model->vertexNormalNumber, cudaMemcpyHostToDevice);


	// 페이스(폴리곤) 데이터 전달
	cudaMemcpy(dev_facePositionIndex,
		model->facePositionIndex.data(),
		sizeof(DEV_Vector4i) * model->faceNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_faceUVIndex,
		model->faceUVIndex.data(),
		sizeof(DEV_Vector4i) * model->faceNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_faceNormalIndex,
		model->faceNormalIndex.data(),
		sizeof(DEV_Vector4i) * model->faceNumber, cudaMemcpyHostToDevice);
	// 텍스쳐 데이터 전달
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
	// GPU 메모리 -> CPU 메모리
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

	// std::cout << "프레임 버퍼 초기화...\n";
}

void Kernel::StartVertexShader(int vertexNumber, int vertexNormalNumber)
{
	// 스레드 블록 나누기
	// 버텍스 하나당 스레드 4개, WARP 32일때 블록 하나에 버텍스 8개
	int gridX = ((vertexNumber * 4) % WARP) == 0 ? vertexNumber * 4 / WARP : (vertexNumber * 4 / WARP) + 1;
	dim3 grid(gridX, 1);
	dim3 block(WARP, 1); // 와프단위로 짜르기

	// MVP 변환 실행
	VertexShader<<<grid, block >>>(dev_vertexPosition, dev_clipSpacePosition, dev_MVP, vertexNumber);
	// 동기화
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "VertexShader cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	gridX = ((vertexNumber * 3) % WARP) == 0 ? vertexNumber * 3 / WARP : (vertexNumber * 3 / WARP) + 1;
	grid.x = gridX;
	
	// 라이트 계산을 위해 노말 월드스페이스 기준으로 바꾸기
	LocalToWolrdNormal<<<grid, block>>>(dev_vertexNormal, dev_worldSpaceNormal, 
		dev_worldSpaceNormalTransform, vertexNormalNumber);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "LocalToWolrdNormal cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// std::cout << "버텍스 셰이더까지는 잘 실행 되었다...\n";
}

void Kernel::StartRasterizer(int vertexNumber, int faceNumber, int frameWidth, int frameHeight)
{
	// 스레드 블록 나누기
	// 버텍스 하나당 스레드 1개, WARP 32일때 블록 하나에 버텍스 32개
	int gridX = vertexNumber % WARP == 0 ? vertexNumber / WARP : (vertexNumber / WARP) + 1;
	dim3 grid(gridX, 1);
	dim3 block(WARP, 1); // 와프단위로 짜르기

	// 원근 나눗셈 부터
	PerspectiveDivision<<<grid, block >>>(dev_clipSpacePosition, vertexNumber);
	// 동기화
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "PerspectiveDivision cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// 클리핑 안 하고 스캔 컨버전에서 자르는 방식으로 하기
	// 페이스 하나당 스레드 한개 WARP 32일때 블록 하나에 페이스 32개
	gridX = faceNumber % WARP == 0 ? faceNumber / WARP : (faceNumber / WARP) + 1;
	grid.x = gridX;
	// 백페이스 컬링

	BackFaceCulling<<<grid, block >>>(dev_facePositionIndex, dev_clipSpacePosition, dev_shouldRenderFace, faceNumber);

	// 동기화
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	    std::cout << "BackFaceCulling cudaDeviceSynchronize fail..." << cudaStatus << "\n";
	    DeleteModelMemory();
	    DeleteOneFrameElements();
	    exit(EXIT_FAILURE);
	}

	// 버텍스 하나당 스레드 4개, WARP 32일때 블록 하나에 버텍스 8개
	gridX = ((vertexNumber * 4) % WARP) == 0 ? vertexNumber * 4 / WARP : (vertexNumber * 4 / WARP) + 1;
	grid.x = gridX;
	// 뷰포트 변환
	ViewportTransform<<<grid, block>>>(dev_clipSpacePosition, dev_viewportSpacePosition, dev_viewportMatrix, vertexNumber);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "ViewportTransform cudaDeviceSynchronize fail..." << cudaStatus << "\n";
		DeleteModelMemory();
		DeleteOneFrameElements();
		exit(EXIT_FAILURE);
	}

	// 스캔 컨버전
	// 페이스 하나당 스레드 한개 WARP 32일때 블록 하나에 페이스 32개
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

	// std::cout << "래스터라이저 문제 없음 \n";
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
	
	// std::cout << "오브젝트 하나 다 그렸다...\n";
}
#pragma endregion
