import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

frames = []
while True:
    ret, frame = cap.read()

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    edge = cv.Canny(frame,75,100)

    edge = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)

    videos = np.hstack((frame, edge))

    cv.imshow('Video display', videos)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()

while(True):
    
    # q 키를 누르고 종료
    if cv.waitKey(1) == ord('q'):
        print('quit')
        cv.destroyAllWindows()
        break
