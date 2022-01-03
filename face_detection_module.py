import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('assets/running.mp4')
previous_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
            bounding_box_c = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bounding_box = int(bounding_box_c.xmin * iw), int(bounding_box_c.ymin * ih), \
                           int(bounding_box_c.width * iw), int(bounding_box_c.height * ih)
            cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(10)
