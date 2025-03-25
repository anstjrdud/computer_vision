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


