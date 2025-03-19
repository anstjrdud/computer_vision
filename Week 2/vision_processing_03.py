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