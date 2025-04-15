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
