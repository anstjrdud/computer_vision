# 1번
* SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현하라.
* YOLOv4와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 추출한다.
* 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적을 유지한다.
* 추적된 각 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오 프레임에 표시하여 실시간으로 출력한다.

## 전체 코드
```python
import cv2 as cv
import numpy as np
from sort import Sort
import time

# YOLO 설정
net = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 비디오 열기
cap = cv.VideoCapture("2165-155327596.mp4")

# SORT 초기화
tracker = Sort()

frame_count = 0
skip_frames = 20  # 20프레임마다 한 번 검출

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    height, width, _ = frame.shape
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # 객체 검출
    for output in detections:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (det[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detections_for_sort = []

    # NMS 결과에서 객체를 추적할 수 있도록 변환
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            detections_for_sort.append([x1, y1, x2, y2, confidences[i]])

    # numpy array로 변환
    detections_for_sort = np.array(detections_for_sort)

    # detections_for_sort가 빈 배열이 아닐 때만 tracker.update 호출
    if detections_for_sort.shape[0] > 0:
        tracks = tracker.update(detections_for_sort)
    else:
        tracks = []

    # 추적 결과 그리기
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        # 객체의 경계 상자 그리기
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 객체의 ID 표시
        cv.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow("YOLOv4 + SORT Tracker", frame)

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```

## 필요 파일
* 2165-1553275986.mp4 : 다중 객체를 추적할 영상
* coco.names : 각 클래시의 이름이 저장되어있는 파일
* sort.py : SORT 알고리즘에 대한 소스 코드
* yolov4.cfg : YOLOv4 모델 파일
* yolov4.weights : YOLOv4 가중치 파일 (여기엔 파일 용량 문제로 올리지 못했습니다. https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights 여기서 받으면 됩니다.)

## 원리
1. YOLOv4 모델을 불러와 레이어를 설정한다.
```python
net = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
```
2. 비디오를 연다.
```python
cap = cv.VideoCapture("2165-155327596.mp4")
```
3. SORT 알고리즘을 초기화한다.
```python
tracker = Sort()
```
4. 프레임을 추출한다. 단, 20프레임에 한 번 검출할 수 있도록 한다.
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    height, width, _ = frame.shape
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
```
5. 각 프레임을 분석해서, 객체를 추출한다.
```python
    for output in detections:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (det[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
```
6. 감지된 객체를 numpy로 변환하여, 추적 결과를 그린다.
```python
    # numpy array로 변환
    detections_for_sort = np.array(detections_for_sort)

    # detections_for_sort가 빈 배열이 아닐 때만 tracker.update 호출
    if detections_for_sort.shape[0] > 0:
        tracks = tracker.update(detections_for_sort)
    else:
        tracks = []

    # 추적 결과 그리기
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        # 객체의 경계 상자 그리기
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 객체의 ID 표시
        cv.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
```
7. 프레임을 출력할 수 있도록 한다.
```python
    cv.imshow("YOLOv4 + SORT Tracker", frame)
```
8. 키 q를 누르면 종료될 수 있도록 한다.
```python
    # 'q' 키를 누르면 종료
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
## 결과
![1번 결과](https://github.com/user-attachments/assets/9956effd-e354-42f5-8095-19370e06f31e)
자동차들이 지나가면서 새로운 객체(자동차)를 실시간으로 인식한다.

# 2번
* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 랜드마크를 추출하고, 이를 실시간 영상으로 시각화하는 프로그램을 구현한다.
* OpenCV를 사용하여 실시간 영상을 캡처한다.
* ESC키를 누르면 프로그램이 종료되도록 한다.

## 전체 코드
```python
import cv2
import mediapipe as mp

# Mediapipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Webcam 캡처
cap = cv2.VideoCapture(0)

# FaceMesh 설정
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 색상 변환 (BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 검출
        results = face_mesh.process(rgb_frame)

        # 랜드마크가 검출되었을 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크 시각화
                for landmark in face_landmarks.landmark:
                    # 정규화된 좌표를 이미지 크기에 맞게 변환
                    h, w, c = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # 랜드마크 점을 원으로 시각화
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # 랜드마크를 연결하여 그리기 (선 연결)
                #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # 영상 출력
        cv2.imshow('FaceMesh', frame)

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 캡처 종료
cap.release()
cv2.destroyAllWindows()
```

## 원리
1. Mediapipe의 FaceMesh를 불러와 초기화한다.
```python
# Mediapipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
```
2. 웹캠으로 영상을 캡처할 수 있게 한다.
```python
cap = cv2.VideoCapture(0)
```
3. FaceMesh의 추적 최소 임계값을 설정한다.
```python
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```
4. 설정된 FaceMesh를 통해서 얼굴 랜드마크를 검출할 수 있도록 한다. (여기선 랜드마크 점을 서로 연결하지 않는다.)
```python
  as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 색상 변환 (BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 검출
        results = face_mesh.process(rgb_frame)

        # 랜드마크가 검출되었을 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크 시각화
                for landmark in face_landmarks.landmark:
                    # 정규화된 좌표를 이미지 크기에 맞게 변환
                    h, w, c = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # 랜드마크 점을 원으로 시각화
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # 랜드마크를 연결하여 그리기 (선 연결)
                #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
```
5. 설정된 FaceMesh로 웹캠을 실행하여 얼굴 랜드마크 영상을 출력한다.
```python
cv2.imshow('FaceMesh', frame)
```
6. ESC 키를 누르면 종료될 수 있도록 한다.
```python
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 캡처 종료
cap.release()
cv2.destroyAllWindows()
```

## 결과
![2번 결과](https://github.com/user-attachments/assets/bc2683c4-f48c-46a5-894a-66382d86b435)
