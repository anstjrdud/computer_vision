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
