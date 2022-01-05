import cv2
import mediapipe as mp
import serial
import time
import math


ser = serial.Serial('COM6', 9600)
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(model_complexity=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
horizontal_distance = 0
vertical_distance = 0;

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                if id == 0:
                    x0 = int(lm.x*w)
                    y0 = int(lm.y*h)
                if id == 5:
                    x5 = int(lm.x*w)
                    y5 = int(lm.y*h)
                if id == 17:
                    x17 = int(lm.x*w)
                    y17 = int(lm.y*h)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            if x17 is not None and x5 is not None and x0 is not None:
                vertical_distance = int(max(0, min(180, math.hypot((x17 + x5)/2 - x0, (y17 + y5)/2 - y0))))
                horizontal_distance = int(max(0, min(180, math.hypot(x17 - x5, y17 - y5))))
                print(vertical_distance, horizontal_distance)
                string_to_send = f'{horizontal_distance},{vertical_distance},\n'
                ser.write(string_to_send.encode("utf-8"))
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
