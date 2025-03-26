import cv2 as cv
import skimage
import numpy as np
from matplotlib import pyplot as plt

# 입력 영상 불러오기
src = cv.imread('coffee cup.jpg')

# 사각형 지정을 통한 초기 분할
mask = np.zeros(src.shape[:2], np.uint8) # 마스크
bgdModel = np.zeros((1, 65), np.float64) # 배경 모델 무조건 1행 65열, float64
fgdModel = np.zeros((1, 65), np.float64) # 전경 모델 무조건 1행 65열, float64

rc = cv.selectROI(src)

# RECT는 사용자가 사각형 지정. 이 값에서 계속 업데이트
cv.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)

# mask 4개 값을 2개로 변환
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

plt.subplot(1, 3, 1)
plt.imshow(src)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask2 * 255)
plt.title('mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(dst)
plt.title('Result')
plt.axis('off')

plt.show()
