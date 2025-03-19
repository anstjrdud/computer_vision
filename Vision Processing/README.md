# Vision_Processing_Basic

## 1번 문제
* 길거리 사진을 불러와 그레이스케일로 변환하고, 특정 임계값을 설정하여 이진화하고 그 이미지의 히스토그램의 계산하고 시각화하라.
### 전체 코드
```python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

h = cv.calcHist([gray],[0],None,[256],[0,256])

plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

plt.plot(h, color='r', linewidth=1), plt.show()

t, bin_img = cv.threshold(img[:,:,2], 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오쥬 알고리즘이 찾느 최적 임곗값=', t)

h = cv.calcHist([bin_img],[0],None,[256],[0,256])

plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

plt.plot(h, color='r', linewidth=1), plt.show()
```
### 원리
1. cv.imread 로 길거리 사진을 불러온다.
```python
img = cv.imread('mistyroad.jpg')
```
2. cv.cvtColor와 cv.COLOR_BGR2GRAY를 이용해서 그레이스케일로 변환한다.
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
3. 우선, 그레이스케일로 변환만 한 이미지를 출력하고, 그 것을 히스토그램으로 시각화한다.
```python
h = cv.calcHist([gray],[0],None,[256],[0,256])
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()
plt.plot(h, color='r', linewidth=1), plt.show()
```
4. cv.thredshold를 통해서 임계값을 정하고, 그 임계값에 따라서 이미지를 이진화한다.
```python
t, bin_img = cv.threshold(img[:,:,2], 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```
5. calcHist를 통해서 히스토그램을 구하고 히스토그램을 시각화한다.
```python
h = cv.calcHist([bin_img],[0],None,[256],[0,256])
```
6. plt.imshow를 통해서 이미지를 출력하고, plt.plot을 통해서 히스토그램을 시각화한다.
```python
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()
plt.plot(h, color='r', linewidth=1), plt.show()
```
### 결과
#### 이진화 전
![Image](https://github.com/user-attachments/assets/2e9968d0-9364-46f7-9ebb-00f44d9eae78)
![Image](https://github.com/user-attachments/assets/ca98d07b-10d7-4c25-984f-5e74f4fd4155)
#### 이진화 후
![Image](https://github.com/user-attachments/assets/dab6664f-bbf0-48e2-90c5-67906cce3648)
![Image](https://github.com/user-attachments/assets/82366f31-4856-464a-a490-25c82850a071)


## 2번 문제
* John Hancocks 간판 이미지를 이진화된 것으로 모폴로지 연산을 적용하여 적용한 이미지들을 한 줄로 나란히 배치하라.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

t, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

# 팽창
d_img = cv.morphologyEx(bin_img, cv.MORPH_DILATE,kernel)

# 침식
e_img = cv.morphologyEx(bin_img, cv.MORPH_ERODE,kernel)

# 열림
o_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN,kernel)

# 닫힘
c_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE,kernel)

# 사진 해상도 조절

X_per = 700/bin_img.shape[1]
bin_img = cv.resize(bin_img, dsize=(0,0), fx=X_per, fy=X_per, interpolation=cv.INTER_AREA)
d_img = cv.resize(d_img, dsize=(0,0), fx=X_per, fy=X_per, interpolation=cv.INTER_AREA)
e_img = cv.resize(e_img, dsize=(0,0), fx=X_per, fy=X_per, interpolation=cv.INTER_AREA)
o_img = cv.resize(o_img, dsize=(0,0), fx=X_per, fy=X_per, interpolation=cv.INTER_AREA)
c_img = cv.resize(c_img, dsize=(0,0), fx=X_per, fy=X_per, interpolation=cv.INTER_AREA)


result = np.hstack((bin_img, d_img, e_img, o_img, c_img))

cv.imshow('result',result)

cv.waitKey()
cv.destroyAllWindows()
```

### 원리
1. cv.imread로 John Hancocks 간판 이미지를 불러온다.
```python
img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)
```
2. cv.threshold를 통해 이미지를 이진화한다.
```python
t, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```
3. cv.getStructuringElement를 통해서 사각형 커널을 만든다.
```python
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
```
4. 모폴로지 연산인 팽창, 침식, 열림, 닫힘을 적용하기 위해서, cv.morphologyEx를 통해서 이미지에 연산을 적용할 수 있도록 한다.
```python
# 팽창
d_img = cv.morphologyEx(bin_img, cv.MORPH_DILATE,kernel)

# 침식
e_img = cv.morphologyEx(bin_img, cv.MORPH_ERODE,kernel)

# 열림
o_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN,kernel)

# 닫힘
c_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE,kernel)
```
5. np.hstack를 통해서 이미지를 한 줄로 나열한다.
```python
result = np.hstack((bin_img, d_img, e_img, o_img, c_img))
```
6. cv.imshow를 통해 결과를 출력한다.
```python
cv.imshow('result',result)
```

### 결과 (순서대로 원본, 팽창, 침식, 열림, 닫힘)
![Image](https://github.com/user-attachments/assets/e06b0d93-5a4c-4110-b8bc-d1b3bb018074)

## 3번 문제
* 나무 이미지를 불러와서 45도 회전하고 회전한 이미지를 1.5배 확대하여 선형 보간을 적용하여 출력하라.
### 전체 코드
```python
import cv2 as cv
import numpy as np

img = cv.imread('tree.png')

imgs = []

cols, rows = img.shape[1], img.shape[0]


cp = (cols // 2,rows // 2)
rot = cv.getRotationMatrix2D(cp,45,1.5)

dst = cv.warpAffine(img,rot,(0,0))

dst = cv.resize(dst,dsize=(0,0),fx=1,fy=1, interpolation=cv.INTER_LINEAR)

imgs = np.hstack((img,dst))

cv.imshow('geometry',imgs)

cv.waitKey()
cv.destroyAllWindows()
```
### 원리
1. cv.imread를 통해서 나무 이미지를 불러온다.
```python
img = cv.imread('tree.png')
```
2. (이미지).shape[1] , (이미지).shape[0] 를 통해 각각 이미지의 가로와 세로 크기를 구한다.
```python
cols, rows = img.shape[1], img.shape[0]
```
3. cv.getRotationMatrix2D를 통해 2번에서 구한 이미지의 크기를 가지고 이미지의 중심점을 찾고, 그 점을 회전점으로 하여 45도 회전하고 크기를 1.5배 확대할 수 있도록 행렬을 생성한다.
```python
cp = (cols // 2,rows // 2)
rot = cv.getRotationMatrix2D(cp,45,1.5)
```
4. cv.warpAffine을 통해 3번에서 생성한 행렬을 가지고 이미지의 회전과 확대를 적용할 수 있도록 한다.
```python
dst = cv.warpAffine(img,rot,(0,0))
```
5. cv.resize를 통해서 INTER_LINEAR를 적용하여 선형 보간을 적용할 수 있도록 한다.
```python
dst = cv.resize(dst,dsize=(0,0),fx=1,fy=1, interpolation=cv.INTER_LINEAR)
```
6. np.hstack로 원본 이미지와 변환 이미지를 나란히 비교할 수 있도록 한다.
```python
imgs = np.hstack((img,dst))
```
7. cv.imshow를 통해 결과를 출력한다.
```python
cv.imshow('geometry',imgs)
```
### 결과
![Image](https://github.com/user-attachments/assets/2fe278c0-7dcc-4ed7-b3f3-bac4fd659254)
