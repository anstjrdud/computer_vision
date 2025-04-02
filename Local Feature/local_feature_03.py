import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 이미지 로드 및 그레이스케일 변환
img1 = cv.imread('img1.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출 및 기술자 계산
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher를 이용한 매칭
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점 선택 (ratio test)
T = 0.4
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# 좌표 추출
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match]).reshape(-1, 1, 2)
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match]).reshape(-1, 1, 2)

# Homography 계산
H, mask = cv.findHomography(points1, points2, cv.RANSAC)

# 이미지 크기
h, w, _ = img2.shape

# 원본 이미지의 테두리를 변환
box1 = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
box2 = cv.perspectiveTransform(box1, H)

# 변환된 이미지의 경계 상자 찾기
x, y, w, h = cv.boundingRect(box2)

# 새로운 직사각형을 위한 box1과 box2 설정
rect_corners = np.float32([[x, y], [x, y+h], [x+w, y+h], [x+w, y]]).reshape(-1, 1, 2)
dst_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

rect_corners = np.where(rect_corners < 0, 0, rect_corners)

# 새로운 변환 행렬 계산
M = cv.getPerspectiveTransform(rect_corners, dst_corners)

# 변환 적용
aligned_img = cv.warpPerspective(img2, M, (w, h))

# **자동 크롭을 적용해 검은 영역 제거**
gray_aligned = cv.cvtColor(aligned_img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray_aligned, 1, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 가장 큰 외곽 영역 찾기
x, y, w, h = cv.boundingRect(contours[0])
aligned_img = aligned_img[y:y+h, x:x+w]  # 크롭

cv.polylines(img2,[np.int32(rect_corners)],True,(0,255,0),8)

# 원래 크기로 맞춤 (비율 유지)
aligned_img = cv.resize(aligned_img, (img2.shape[1], img2.shape[0]), interpolation=cv.INTER_LINEAR)

# 매칭된 점들 시각화
img_match = np.empty((h, w + w, 3), dtype=np.uint8)
dst = cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.imshow(dst)
plt.title('Original')
plt.axis('off')

aligned_img = cv.drawMatches(img1, kp1, aligned_img, kp2, None, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.subplot(2, 1, 2)
plt.imshow(aligned_img)
plt.title('Result')
plt.axis('off')

plt.show()


