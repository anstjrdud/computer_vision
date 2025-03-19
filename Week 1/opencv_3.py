import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg')


if img is None:
    sys.exit('파일이 존재하지 않습니다.')

def draw(event,x,y,flags,param):
    global ix,iy,image

    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP:
        cv.rectangle(img, (ix,iy), (x,y), (0,0,255), 2)
        roi = img[iy:y, ix:x]
        img2 = roi.copy()
        image = roi.copy()
        cv.imshow('roi',img2)

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    
    # q 키를 누르고 종료
    if cv.waitKey(1) == ord('q') or cv.waitKey(1) == ord('Q'):
        print('quit')
        cv.destroyAllWindows()
        break

    # r 키를 누르고 영역 선택을 리셋하고 처음부터 다시 선택
    elif cv.waitKey(1) == ord('r') or cv.waitKey(1) == ord('R'):
        print('reset complete')
        cv.destroyAllWindows()
        img = cv.imread('soccer.jpg')
        cv.imshow('Drawing', img)
        cv.setMouseCallback('Drawing', draw)

    # s 키를 누르고 선택한 영역을 이미지 파일로 저장
    elif cv.waitKey(1) == ord('s') or cv.waitKey(1) == ord('S'):
        print('save complete')
        cv.imwrite('saved_image.jpg',image)
