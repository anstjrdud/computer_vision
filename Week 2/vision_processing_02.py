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
