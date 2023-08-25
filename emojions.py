import cv2

cv2.namedWindow("test")
vc = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # load face classifier

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv2.flip(frame, 1)
else:
    rval = False

while rval:
    cv2.imshow("preview", frame) # show current frame

    # read and flip frame from the video capture
    rval, frame = vc.read()
    frame = cv2.flip(frame, 1)

    # find a face in the image
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) # find a face in the image.
    # face is a tuple with x, y, width and height; extract
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
