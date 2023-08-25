from cv2 import CascadeClassifier, data, resize
import numpy as np

class FaceCropper:

    def __init__(self):
        # parameters for face detection
        self.face_classifier: CascadeClassifier = \
            CascadeClassifier(data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scale_factor = 1.1
        self.min_neighbours = 5
        self.min_size = (48, 48)

        # parameters for image cropping
        self.face_img_size = 220
        
    def get_face(self, frame):
        """
        Returns (bool, Rect): boolean value for whether any face was found, and bounding Rect
        for the largest face found.
        Please ensure that the frame is in greyscale before sending through!
        """
        faces = self.face_classifier.detectMultiScale(
            frame, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbours, 
            minSize=self.min_size)
        
        # Get the largest face rect and return only that
        biggest_face = None
        face_detected = False
        for face in faces:
            if not face_detected:
                biggest_face = face
                face_detected = True
            else:
                # compare face areas
                _, _, w, h = face
                _, _, bfw, bfh = biggest_face
                if w * h > bfw * bfh:
                    biggest_face = face
        
        return face_detected, biggest_face

    def get_face_img(self, frame, face_rect):
        """Please ensure that the frame is in greyscale before sending through!"""
        x, y, w, h = face_rect
        frame_crop = frame[y:y + h, x:x + w]
        # Must crop to 48x48 to match the emotion recognition dataset
        face_img = np.copy(resize(frame_crop, (48, 48)))
        return face_img
