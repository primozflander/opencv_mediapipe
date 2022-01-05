import cv2
import mediapipe as mp
import time


mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

cap = cv2.VideoCapture('assets/running.mp4')
previous_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL, draw_spec, draw_spec)
            for landmark in face_landmarks.landmark:
                # print(landmark)
                ih, iw, ic = img.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                print(x, y)
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(10)
