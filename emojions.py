from cv2 import flip, imshow, rectangle, waitKey, namedWindow, VideoCapture, destroyWindow, cvtColor, COLOR_BGR2GRAY
from utils.face_crop import FaceCropper

namedWindow("test")
vc = VideoCapture(0)

face_cropper: FaceCropper = FaceCropper()

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    image = flip(frame, 1)
else:
    rval = False

while rval:
    imshow("preview", image) # show current frame

    # read and flip frame from the video capture
    rval, frame = vc.read()
    image = flip(frame, 1)
    gray_image = cvtColor(image, COLOR_BGR2GRAY)

    # face is a tuple with x, y, width and height; extract
    face_detected, face = face_cropper.get_face(gray_image)
    if face_detected:
        # x, y, w, h = face
        # rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        image = face_cropper.get_face_img(gray_image, face)
    else:
        image = frame

    key = waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
destroyWindow("preview")
