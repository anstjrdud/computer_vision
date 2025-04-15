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
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# SORT 초기화
tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
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



