from utils.face_crop import FaceCropper
from cv2 import flip, imshow, rectangle, waitKey, namedWindow, VideoCapture, destroyWindow, cvtColor, COLOR_BGR2GRAY, putText, FONT_HERSHEY_DUPLEX, LINE_AA

import numpy as np

def main():
    face_cropper: FaceCropper = FaceCropper()
    namedWindow('Emojions')
    vc = VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        image = flip(frame, 1)
    else:
        rval = False

    while rval:
        imshow("Emojions", image) # show current frame

        # read and flip frame from the video capture
        rval, frame = vc.read()
        if not rval:
            break
        image = flip(frame, 1)
        gray_image = cvtColor(image, COLOR_BGR2GRAY)

        # face is a tuple with x, y, width and height; extract
        face_detected, face = face_cropper.get_face(gray_image)
        if face_detected:
            x, y, w, h = face
            
            # Retrieve an image of correct proportions and pass through emotion detection model
            # face_image = face_cropper.get_face_img_for_emotion(gray_image, face)

            # Display emotion and emoji; put rect around face
            pad = int(np.floor(w / 2))
            rectangle(image, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 4)

        key = waitKey(20)
        if key == 27: # exit on ESC
            vc.release()
            destroyWindow('Emojions')
            waitKey(1)

if __name__=="__main__":
    main()