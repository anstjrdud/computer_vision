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