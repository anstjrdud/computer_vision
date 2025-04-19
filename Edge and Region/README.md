# 1번
* 주어진 이미지를 통해 그레이스케일로 변환한다.
* 소벨 필터로 X축과 Y축의 에지를 검출한다.
* 검출된 에지 강도 이미지를 시각화한다.
* 원본 이미지와 에지 강도 이미지를 나란히 시각화한다.

## 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_64F,1,0,ksize=3)
grad_y = cv.Sobel(gray, cv.CV_64F,0,1,ksize=3)

magni = cv.magnitude(grad_x,grad_y)

edge_strength = cv.convertScaleAbs(magni)


plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap='gray')
plt.title('Edge Strength')
plt.axis('off')

plt.show()
```

## 함수
### cv2.Sobel
```python 
cv2.Sobel (src, depth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None)
```

X축과 Y축 방향의 에지를 검출할 함수.
* src : 입력 영상
* depth : 출력 영상 데이터 타입. -1이면 입력 영상과 같은 데이터 타입을 사용.
* dx : x 방향 미분 차수. 1차미분할지 2차미분 할지 결정
* dy : y 방향 미분 차수.
* dst : 출력 영상(행렬)
* ksize : 커널 크기. 기본값은 3.
* scale : 연산 결과에 추가적으로 곱할 값. 기본값은 1.
* delta : 연산 결과에 추가적으로 더할 값. 기본값은 0.
* borderType : 가장자리 픽셀 확장 방식. 기본값은 cv2.BORDER_DEFAULT.

### cv2.magnitude
```python 
cv2.magnitude(x, y, magnitude=None)
```

검출된 에지의 강도를 계산하는 함수.
* x : 2D 벡터의 x 좌표 행렬. 실수형.
* y : 2D 벡터의 y 좌표 행렬. x와 같은 크기. 실수형
* magnitude : 2D 벡터의 크기 행렬. x와 같은 크기, 같은 타입.

### cv2.convertScaleAbs
```python
cv2.convertScaleAbs(src)
 ```

에지 강도 이미지의 데이터 타입을 변환하는 함수.
* src : uint8로 변환할 에지 강도 이미지

## 결과
![1번 결과](https://github.com/user-attachments/assets/1a2751b8-f6da-4fda-a09c-07551fce1979)

# 2번
* 주어진 이미지를 가지고 캐니(Canny) 에지 검출을 사용하여 에지 맵을 생성한다.
* 허프 변환(Hough Transform)을 사용하여 이미지에서 직선을 검출한다.
* 검출된 직선을 원본 이미지에 빨간색으로 표시한다.
* 원본 이미지와 직선이 그려진 이미지를 나란히 시각화한다.

## 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('coffee cup.jpg')
img_result = cv.imread('coffee cup.jpg')

gray = cv.cvtColor(img_result, cv.COLOR_BGR2GRAY)

canny1 = cv.Canny(gray,100,200)

lines = cv.HoughLinesP(canny1,10,np.pi/180.,100,minLineLength=7,maxLineGap=12)

if lines is not None: # 라인 정보를 받았으면
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1]) # 시작점 좌표 x,y
        pt2 = (lines[i][0][2], lines[i][0][3]) # 끝점 좌표, 가운데는 무조건 0
        cv.line(img_result, pt1, pt2,(255, 0, 0), 2, cv.LINE_AA)


plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_result)
plt.title('Result')
plt.axis('off')

plt.show()

```

## 함수
### cv2.Canny
```python 
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
```
최소, 최대 임계값을 정해서 이미지의 엣지를 검출하는 함수.
* image : 입력 이미지.
* threshold1 : 엣지 검출에서 사용되는 최소 임계값. 이 값보다 낮으면 엣지로 간주 하지 않음.
* threshold2 : 엣지 검출에서 사용되는 최대 임계값. 이 값보다 높으면 확실한 엣지로 간주.
* edges : 선택적으로 출력되는 엣지 이미지.
* apertureSize : 소벨 연산자에 사용되는 커널 크기를 지정.
* L2gradient : 그라디언트(Gradient) 크기를 계산할 때 사용할 방법을 지정.

### cv2.HoughLinesP
```python 
cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
```
확률적인 허프 변환으로 이미지에서 직선을 검출하는 함수.
* image : 입력 에지 이미지.
* rho : 축적 배열에서 rho 값의 간격. ex) 1.0 -> 1픽셀 간격
* theta : 축적 배열에서 theta 값의 간격, 𝜃 값의 범위.
* threshold: 축적 배열에서 직선으로 판단할 임계값.
* lines: 선분의 시작과 끝 좌표(x1, y1, x2, y2) 정보를 담고 있는 numpy.ndarray.
* minLineLength: 검출할 선분의 최소 길이.
* maxLineGap: 직선으로 간주할 최대 에지 점 간격.

### cv2.line
```python
cv2.line(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
```
검출된 직선을 이미지에 그리는 함수.
* img : 선을 그릴 이미지.
* pt1, pt2 : 직선의 시작점과 끝점을 나타낼 튜플 (x,y)
* color : 선 색상 또는 밝기 (R, G, B) 튜플 또는 정수값
* thickness : 선 두께. 기본 값은 1.
* lineType : 선 타입. cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA 중 선택.
* shift : 그리기 좌표 값의 축소 비율. 기본값은 0.

## 결과
![2번 결과](https://github.com/user-attachments/assets/a910b9c6-ed82-41f0-ae4e-bf1830b7a4d3)

# 3번
* 주어진 이미지에서 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 통해 객체를 추출한다.
* 객체 추출 결과를 마스크 형태로 시각화한다.
* 원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력한다.
* 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화 한다.

## 전체 코드
```python
import cv2 as cv
import skimage
import numpy as np
from matplotlib import pyplot as plt

# 입력 영상 불러오기
src = cv.imread('coffee cup.jpg')

# 사각형 지정을 통한 초기 분할
mask = np.zeros(src.shape[:2], np.uint8) # 마스크
bgdModel = np.zeros((1, 65), np.float64) # 배경 모델 무조건 1행 65열, float64
fgdModel = np.zeros((1, 65), np.float64) # 전경 모델 무조건 1행 65열, float64

rc = cv.selectROI(src)

# RECT는 사용자가 사각형 지정. 이 값에서 계속 업데이트
cv.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)

# mask 4개 값을 2개로 변환
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

plt.subplot(1, 3, 1)
plt.imshow(src)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask2 * 255)
plt.title('mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(dst)
plt.title('Result')
plt.axis('off')

plt.show()
```

## 함수
### cv2.grabCut
```python 
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
```
대화식 분할을 수행하는 함수.
* image : 입력 이미지.
* mask : 입출력 마스크.
* rect : ROI 영역. cv2.GC_INIT_WITH_RECT 모드에서만 사용됨
* bgdModel: 임시 배경 모델 행렬. 같은 영상 처리 시에는 변경 금지.
* fgdModel: 임시 전경 모델 행렬. 같은 영상 처리 시에는 변경 금지.
* iterCount: 결과 생성을 위한 반복 횟수.
* mode: cv2.GC_로 시작하는 모드 상수. 보통 cv2.GC_INIT_WITH_RECT 모드로 초기화하고, cv2.GC_INIT_WITH_MASK 모드로 업데이트함.

### np.where
```python 
np.where((조건문), (True), (False))
```
조건문에 따라서 배열의 값을 바꾸는 연산을 실행하는 함수
* 조건문 : 배열의 값에 대한 조건
* True : 조건문을 만족할 경우에 실행되는 연산
* False : 조건문을 만족하지 않을 경우에 실행되는 연산

## 결과
![3번 결과](https://github.com/user-attachments/assets/4026d808-0fb3-4a55-ad0e-1d7e2fd7b183)
(본 결과는 지정한 사각형 영역에 따라서 달라질 수 있음.)






