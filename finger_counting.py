import cv2
import time
import os
import hand_tracking_module as htm


wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
cTime = 0


folder_path = "assets/hand_images"
file_list = os.listdir(folder_path)
# print(file_list)
overlay_list = []
for image_path in file_list:
    image = cv2.imread(f'{folder_path}/{image_path}')
    overlay_list.append(image)
    #print(f'{folder_path}/{image_path}')
detector = htm.HandDetector(detectionConfidence=0.7)
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers_state = []
        if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1]:
            # print("index finger open")
            fingers_state.append(1)
        else:
            # print("index finger closed")
            fingers_state.append(0)
        for id in range(1, 5):
            if lmList[tip_ids[id]][2] < lmList[tip_ids[id]-2][2]:
                # print("index finger open")
                fingers_state.append(1)
            else:
                # print("index finger closed")
                fingers_state.append(0)
        # print(fingers_state)
        total_fingers = fingers_state.count(1)
        # print(total_fingers)
        h, w, c = overlay_list[total_fingers].shape
        img[0:h, 0:w] = overlay_list[total_fingers]
        cv2.rectangle(img, (20, 300), (110, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
