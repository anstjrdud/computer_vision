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
