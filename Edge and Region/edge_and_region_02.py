import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('dabotop.jpg')
img_result = cv.imread('dabotop.jpg')

gray = cv.cvtColor(img_result, cv.COLOR_BGR2GRAY)

canny1 = cv.Canny(gray,100,200)

lines = cv.HoughLinesP(canny1,10,np.pi/180.,120,minLineLength=15,maxLineGap=5)

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
