import cv2
import mediapipe as mp
import time


class FaceDetector:

    def __init__(self, det_confidence=0.5):
        self.det_confidence=det_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.det_confidence)

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bounding_boxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id, detection)
                bounding_box_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bounding_box = int(bounding_box_c.xmin * iw), int(bounding_box_c.ymin * ih), \
                               int(bounding_box_c.width * iw), int(bounding_box_c.height * ih)
                bounding_boxes.append([bounding_box, detection.score])
                # cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
                if draw:
                    img = self.advanced_draw(img, bounding_box)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        return img, bounding_boxes

    def advanced_draw(self, img, bounding_box, l=30, t=5):
        x, y, w, h = bounding_box
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bounding_box, (255, 0, 255), 1)
        # top left
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # top right
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # bottom left
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        # bottom right
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)
        return img


def main():
    cap = cv2.VideoCapture('assets/running.mp4')
    previous_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bounding_boxes = detector.find_faces(img)
        # print(bounding_boxes)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()