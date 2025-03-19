# Vision_Processing_Basic

1번 문제
* 길거리 사진을 불러와 그레이스케일로 변환하고, 특정 임계값을 설정하여 이진화하고 그 이미지의 히스토그램의 계산하고 시각화하라.

원리
1. cv.imread 로 길거리 사진을 불러온다.
2. cv.cvtColor와 cv.COLOR_BGR2GRAY를 이용해서 그레이스케일로 변환한다.
3. cv.thredshold를 통해서 임계값을 정하고, 그 임계값에 따라서 이미지를 이진화한다.
4. cv.equalizeHist를 통해서 히스토그램을 평활화하고 이미지를 출력한다.
5. calcHist를 통해서 히스토그램을 구하고 히스토그램을 시각화한다.

결과

![Image](https://github.com/user-attachments/assets/faafaef2-a1af-4d7d-bb82-b85bf88a5138)
![Image](https://github.com/user-attachments/assets/3b355dac-e84d-451e-ac87-00d5f737bdcc)


2번 문제
* John Hancocks 간판 이미지를 이진화된 것으로 모폴로지 연산을 적용하여 적용한 이미지들을 한 줄로 나란히 배치하라.

원리
1. cv.imread로 John Hancocks 간판 이미지를 불러온다.
2. cv.threshold를 통해 이미지를 이진화한다.
3. cv.getStructuringElement를 통해서 사각형 커널을 만든다.
4. 모폴로지 연산인 팽창, 침식, 열림, 닫힘을 적용하기 위해서, cv.morphologyEx를 통해서 이미지에 연산을 적용할 수 있도록 한다.
5. np.hstack를 통해서 이미지를 한 줄로 나열한다.
6. cv.imshow를 통해 결과를 출력한다.

결과 (순서대로 원본, 팽창, 침식, 열림, 닫힘)
![Image](https://github.com/user-attachments/assets/e06b0d93-5a4c-4110-b8bc-d1b3bb018074)

3번 문제
* 나무 이미지를 불러와서 45도 회전하고 회전한 이미지를 1.5배 확대하여 선형 보간을 적용하여 출력하라.

원리
1. cv.imread를 통해서 나무 이미지를 불러온다.
2. (이미지).shape[1] , (이미지).shape[0] 를 통해 각각 이미지의 가로와 세로 크기를 구한다.
3. cv.getRotationMatrix2D를 통해 2번에서 구한 이미지의 크기를 가지고 이미지의 중심점을 찾고, 그 점을 회전점으로 하여 45도 회전하고 크기를 1.5배 확대할 수 있도록 행렬을 생성한다.
4. cv.warpAffine을 통해 3번에서 생성한 행렬을 가지고 이미지의 회전과 확대를 적용할 수 있도록 한다.
5. cv.resize를 통해서 INTER_LINEAR를 적용하여 선형 보간을 적용할 수 있도록 한다.
6. np.hstack로 원본 이미지와 변환 이미지를 나란히 비교할 수 있도록 한다.
7. cv.imshow를 통해 결과를 출력한다.

결과
![Image](https://github.com/user-attachments/assets/2fe278c0-7dcc-4ed7-b3f3-bac4fd659254)
