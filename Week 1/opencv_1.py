import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg')

#이미지지 크기 조절을 위한 resize

X_per = 900/img.shape[1]
img = cv.resize(img, dsize=(0,0), fx=X_per, fy=X_per, interpolation=cv.INTER_AREA)


if img is None:
    sys.exit('파일이 존재하지 않습니다.')

frames = []

# 그레이스케일 이미지 생성
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3차원 배열로 변환 (컬러 채널을 추가하여 img와 같은 차원으로 만듦)
gray_3d = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 원본 이미지와 그레이스케일 이미지를 프레임 리스트에 추가
frames.append(img)
frames.append(gray_3d)

# 이미지가 존재할 때 np.hstack으로 결합
if len(frames) > 0:
    imgs = frames[0]
    imgs = np.hstack((imgs, frames[1]))  
    cv.imshow('soccer images', imgs)

    cv.waitKey()
    cv.destroyAllWindows()

print(type(img))
print(img.shape)
