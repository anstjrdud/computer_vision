import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

t, bin_img = cv.threshold(img[:,:,2], 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오쥬 알고리즘이 찾느 최적 임곗값=', t)

equal = cv.equalizeHist(bin_img)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([equal],[0],None,[256],[0,256])
plt.plot(h, color='r', linewidth=1), plt.show()
