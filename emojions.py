from cv2 import flip, imshow, rectangle, waitKey, namedWindow, VideoCapture, destroyWindow, cvtColor, COLOR_BGR2GRAY, putText, FONT_HERSHEY_DUPLEX, LINE_AA
from utils.face_crop import FaceCropper
from models.emotion_model import EmotionModel

def main():
    namedWindow("test")
    vc = VideoCapture(0)

    face_cropper: FaceCropper = FaceCropper()
    ed_model = EmotionModel(which=1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        image = flip(frame, 1)
    else:
        rval = False

    while rval:
        imshow("preview", image) # show current frame

        # read and flip frame from the video capture
        rval, frame = vc.read()
        if not rval:
            break
        image = flip(frame, 1)
        gray_image = cvtColor(image, COLOR_BGR2GRAY)

        # face is a tuple with x, y, width and height; extract
        face_detected, face = face_cropper.get_face(gray_image)
        if face_detected:
            # Put rectangle around face on the display
            x, y, w, h = face
            rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Retrieve an image of correct proportions and pass through emotion detection model
            face_image = face_cropper.get_face_img(gray_image, face)
            prediction = ed_model.get_prediction(face_image)
            putText(image, prediction, (x, y), FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, LINE_AA)

        key = waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    destroyWindow("preview")

if __name__ == "__main__":
    main()
