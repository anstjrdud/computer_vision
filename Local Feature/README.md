# 1번
* 이미지(mot_color70.jpg)를 불러와 SIFT 객체를 생성한다.
* 그 이미지로부터 특징점을 검출하고 특징점을 이미지에 시각화한다.
* matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력한다.

## 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("mot_color70.jpg")

sift = cv.SIFT_create(nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.06, edgeThreshold = 18, sigma = 1.6)
kp, des = sift.detectAndCompute(img, None)

result = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title('SIFT')
plt.axis('off')

plt.show()
```

## 함수
### cv.SIFT_create
```python
cv.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
```
SIFT 객체를 생성하는 함수
* nfeatures: 검출 최대 특징 수
* nOctaveLayers: 이미지 피라미드에 사용할 계층 수
* contrastThreshold: 필터링할 빈약한 특징 임계값
* edgeThreshold: 필터링할 엣지 임계값 -> 
* sigma: 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값
### (SIFT 객체).detectAndCompute
```python
(SIFT 객체).detectAndCompute(image, mask, decriptors, useProvidedKeypoints)
```
SIFT 객체를 통해 이미지의 특징점을 검출하는 함수
* image: 입력 이미지
* keypoints: 디스크립터 계산을 위해 사용할 특징점
* descriptors(optional): 계산된 디스크립터
* mask(optional): 특징점 검출에 사용할 마스크
* useProvidedKeypoints(optional): True인 경우 특징점 검출을 수행하지 않음

### cv.drawKeypoints
```python
cv.drawKeypoints(img, keypoints, outImg, color, flags)
```
검출한 특징점을 이미지에 시각화하는 함수
* img : 입력 이미지
* keypoints : 표시할 특징점
* outImg : 특징점이 그려질 결과 이미지
* color : 특징점의 색상
* flags : 특징점 표시 방법
-> (cv2.DRAW_MATCHES_FLAGS_DEFAULT: 좌표 중심에 동그라미만 그림(default),
  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 동그라미의 크기를 size와 angle을 반영해서 그림)

## 결과
![1번 결과](https://github.com/user-attachments/assets/a59c42ef-92e3-42a6-a79b-1dd02f01782f)


## 특징점 해설
![1번 해설](https://github.com/user-attachments/assets/99eeca03-1e84-488b-acc7-77445eb3152c)
* T : 두 기술자의 거리가 T보다 작으면 매칭되었다고 간주한다.
이 특징점들은 특징점의 위치, 특징점의 영향 범위에 대한 반경, 회전시 특징점을 식별할 수 있는 각도값으로써의 방향을 알 수 있게한다.

# 2번
* 두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화해라.

## 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img1 = cv.imread('mot_color70.jpg')
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

FLANN_INDEX_KDTREE = 1
index = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index, search_params)
matches = flann.knnMatch(des1, des2, k = 2)

T = 0.55
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype = np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_result = cv.cvtColor(img_match, cv.COLOR_BGR2RGB)


plt.imshow(img_result)
plt.title('Matching result')
plt.axis('off')

plt.show()
```
## 함수
### cv.FlannBasedMatcher
```python
cv.FlannBasedMatcher(index_params, search_params)
```
* index_params : 사전 사료로 생성되며, 어떤 알고리즘을 통해서 특징점을 매칭할 것인지 정한다.
* search_params : 특성 매칭을 위한 반복 횟수
  
### knnMatch
```python
cv.(FlannBasedMatcher).knnMatch(queryDescriptors, trainDescriptors, k, mask=None, compactResult=None)
```
* queryDescriptors: (기준 영상 특징점) 질의 기술자 행렬
* trainDescriptors: (대상 영상 특징점) 학습 기술자 행렬
* k: 질의 기술자에 대해 검출할 매칭 개수
* mask: 매칭 수행 여부를 지정하는 행렬 마스크
* compactResult: mask가 None이 아닐 때 사용되는 파라미터. 기본값은 False이며, 이 경우 결과 matches는 기준 영상 특징점과 같은 크기를 가짐.
* matches: 매칭 결과. cv2.DMatch 객체의 리스트의 리스트

### cv.drawMatches
```python
cv.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor=None, singlePointColor=None, matchesMask=None, flags=None)
```
* img1, keypoints1: 기준 영상과 기준 영상에서 추출한 특징점 정보
* img2, keypoints2: 대상 영상과 대상 영상에서 추출한 특징점 정보
* matches1to2: 매칭 정보. cv2.DMatch의 리스트.
* outImg: 출력 영상 (None)
* matchColor: 매칭된 특징점과 직선 색상, 랜덤한 색상
* singlePointColor: 매칭되지 않은 특징점 색상
* matchesMask: 매칭 정보를 선택하여 그릴 때 사용할 마스크
* flags: 매칭 정보 그리기 방법. 기본값은 cv2.DRAW_MATCHES_FLAGS_DEFAULT.

## 결과
![2번 결과](https://github.com/user-attachments/assets/bbf39462-9746-4087-857b-4c4508762e15)

# 3번
* SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬하세요.

## 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 이미지 로드 및 그레이스케일 변환
img1 = cv.imread('img1.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출 및 기술자 계산
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher를 이용한 매칭
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점 선택 (ratio test)
T = 0.4
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# 좌표 추출
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match]).reshape(-1, 1, 2)
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match]).reshape(-1, 1, 2)

# Homography 계산
H, mask = cv.findHomography(points1, points2, cv.RANSAC)

# 이미지 크기
h, w, _ = img2.shape

# 원본 이미지의 테두리를 변환
box1 = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
box2 = cv.perspectiveTransform(box1, H)

# 변환된 이미지의 경계 상자 찾기
x, y, w, h = cv.boundingRect(box2)

# 새로운 직사각형을 위한 box1과 box2 설정
rect_corners = np.float32([[x, y], [x, y+h], [x+w, y+h], [x+w, y]]).reshape(-1, 1, 2)
dst_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

rect_corners = np.where(rect_corners < 0, 0, rect_corners)

# 새로운 변환 행렬 계산
M = cv.getPerspectiveTransform(rect_corners, dst_corners)

# 변환 적용
aligned_img = cv.warpPerspective(img2, M, (w, h))

# **자동 크롭을 적용해 검은 영역 제거**
gray_aligned = cv.cvtColor(aligned_img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray_aligned, 1, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 가장 큰 외곽 영역 찾기
x, y, w, h = cv.boundingRect(contours[0])
aligned_img = aligned_img[y:y+h, x:x+w]  # 크롭

cv.polylines(img2,[np.int32(rect_corners)],True,(0,255,0),8)

# 원래 크기로 맞춤 (비율 유지)
aligned_img = cv.resize(aligned_img, (img2.shape[1], img2.shape[0]), interpolation=cv.INTER_LINEAR)

# 매칭된 점들 시각화
img_match = np.empty((h, w + w, 3), dtype=np.uint8)
dst = cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.imshow(dst)
plt.title('Original')
plt.axis('off')

aligned_img = cv.drawMatches(img1, kp1, aligned_img, kp2, None, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.subplot(2, 1, 2)
plt.imshow(aligned_img)
plt.title('Result')
plt.axis('off')

plt.show()
```
## 함수
### 
## 결과
![3번 결과](https://github.com/user-attachments/assets/4113ba4c-8610-4ba2-bc0b-faae13528e4b)
