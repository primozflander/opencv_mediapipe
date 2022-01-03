import cv2
import mediapipe as mp

cap = cv2.VideoCapture()

while True:
    success, img = cap.read()
