from cv2 import CascadeClassifier, data, cvtColor, COLOR_BGR2GRAY, Rect

class FaceCropper:

    def __init__(self):
        self.face_classifier: CascadeClassifier = \
            CascadeClassifier(data.haarcascades + "haarcascade_frontalface_default.xml")
        
    def get_face(self, frame):
        """Returns (bool, Rect): boolean value for whether any face was found, and bounding Rect
        for the largest face found."""
        gray_image = cvtColor(frame, COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
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
        pass
