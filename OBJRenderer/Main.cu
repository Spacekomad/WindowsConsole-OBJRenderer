#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <windows.h>
#include <iostream>
#include <conio.h>

#include "ConsoleMonitor.h"
#include "ForwardRenderer.h"

int main(int argc, char** argv)
{
    // 화면 배경 색 지정
    system("color 08");

    // Cuda Device 세팅
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice fail...\n");
        exit(EXIT_FAILURE);
    }

    // 콘솔 모니터 객체 생성. 순서대로
    // 렌더러(렌더링 방식 정의), 콘솔 너비, 콘솔 높이, 디스플레이 픽셀 너비, 디스플레이 픽셀 높이
    ConsoleMonitor gameMonitor(std::make_shared<ForwardRenderer>(), 900, 300, 300, 100);

    // 렌더링 할 씬(맵) 생성
    Scene sampleScene;


    std::shared_ptr<GameObject> TestModel;

    std::shared_ptr<GameObject> tower;

    // 집 모델 게임 오브젝트 생성
    std::shared_ptr<GameObject> house;

    char selectModel = 0;

    while (selectModel < '0' || selectModel > '2') {
        selectModel = _getch();

        // 모델의 페이스는 모두 삼각형으로 잘라서 로딩
        switch (selectModel) {
        case '0':
            TestModel = sampleScene.CreateGameObject("../Models/TestModel.obj", "../Models/Textures/cottage_diffuse.png");
            TestModel->transform.SetPosition(-10, 0, -5);
            break;
        case '1':
            TestModel = sampleScene.CreateGameObject("../Models/Dice.obj", "../Models/Textures/Dice.png");
            TestModel->transform.SetPosition(-10, 0, -5);
            break;
        case '2':
            house = sampleScene.CreateGameObject("../Models/cottage_obj.obj", "../Models/Textures/cottage_diffuse.png");
            // y축 45도 회전
            house->transform.SetRotation(0, 45, 0);
            // 모델정보 텍스트로 출력 (디버깅 용)
            //house->model.front()->PrintModelInfoToText();
            break;
        }
    }


    char input = 0;
    Vector3f position = sampleScene.mainCamera->transform.GetPosition();
    Vector3f rotation = sampleScene.mainCamera->transform.GetRotation();

    while (input != 'm') {
        gameMonitor.RenderScene(sampleScene);

        input = _getch();

        switch (input) {
        case 'a':
            position.x -= 10;
            break;
        case 'd':
            position.x += 10;
            break;
        case 'w':
            position.z += 10;
            break;
        case 's':
            position.z -= 10;
            break;
        case 'p':
            position.y += 10;
            break;
        case 'o':
            position.y -= 10;
            break;
        case 'q':
            rotation.y += 15;
            break;
        case 'e':
            rotation.y -= 15;
            break;
        case 'm':

            break;
        default:
            break;
        }
        sampleScene.mainCamera->transform.SetPosition(position);
        sampleScene.mainCamera->transform.SetRotation(rotation);
    }

    // 2D 이미지 픽셀로 렌더링 예제

    /*ConsoleMonitor imageEx(900, 300, 300, 100);
    imageEx.SetImageToPixels("../Images/img1.png");
    imageEx.DrawDisplayGray();

    getchar();
    imageEx.DrawDisplay();*/

    getchar();

    return 0;
}