from cv2 import CascadeClassifier, data, resize, cvtColor, COLOR_GRAY2RGB
import numpy as np

class FaceCropper:

    def __init__(self):
        # parameters for face detection
        self.face_classifier: CascadeClassifier = \
            CascadeClassifier(data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scale_factor = 1.1
        self.min_neighbours = 5
        self.min_size = (20, 20)

        # parameters for image cropping
        # self.face_img_size = (180, 180)
        self.face_img_size = (256, 256)
        
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

    def get_face_img_for_emotion(self, frame, face_rect):
        """Please ensure that the frame is in greyscale before sending through!"""
        x, y, w, h = face_rect
        frame_crop = frame[y:y + h, x:x + w]
        # Must crop to 48x48 to match the emotion recognition dataset
        face_img = np.expand_dims(np.expand_dims(resize(frame_crop, (48, 48)), -1), 0)
        return face_img
    
    def get_face_img_for_emoji(self, frame, face_rect):
        """Please ensure that the frame is in greyscale before sending through!"""
        x, y, w, h = face_rect
        pad = int(np.floor(w / 4))
        frame = cvtColor(frame, COLOR_GRAY2RGB)
        frame_crop = frame[y-pad:y + h + pad, x - pad:x + w + pad]
        result = np.expand_dims(np.expand_dims(resize(frame_crop, self.face_img_size), -1), 0)
        #result = resize(frame_crop, self.face_img_size)
        return result
