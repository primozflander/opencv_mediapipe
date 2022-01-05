import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_image_mode,
                                                    max_num_faces=self.max_num_faces,
                                                    refine_landmarks=self.refine_landmarks,
                                                    min_detection_confidence=self.min_detection_confidence,
                                                    min_tracking_confidence=self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def find_face_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_landmarks,
                                                self.mp_face_mesh.FACEMESH_FACE_OVAL,
                                                self.draw_spec,
                                                self.draw_spec)
                face = []
                for landmark in face_landmarks.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture('assets/running.mp4')
    previous_time = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(len(faces))
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
