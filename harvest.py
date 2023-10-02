from models.emoji_model import EmojiModel
from utils.face_crop import FaceCropper
from utils.misc import get_new_file_counter
from cv2 import flip, imshow, rectangle, waitKey, namedWindow, VideoCapture, destroyWindow, cvtColor, COLOR_BGR2GRAY, imwrite, imread

import numpy as np
import time
import os

from argparse import ArgumentParser

def main():
    """
    For the purposes of obtaining images for transfer learning
    """
    argparser = ArgumentParser("harvest")
    argparser.add_argument("emoji")
    args = argparser.parse_args()
    emoji = args.emoji

    face_cropper: FaceCropper = FaceCropper()

    namedWindow('Emojions')
    vc = VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        image = flip(frame, 1)
    else:
        rval = False

    prev = time.time()
    start = time.time()

    # initialise counter to the number of images currently in the given emoji dir
    counter = get_new_file_counter(emoji)

    while rval:
        imshow("Emojions", image) # show current frame

        # read and flip frame from the video capture
        rval, frame = vc.read()
        if not rval:
            break
        image = flip(frame, 1)
        frame = image.copy()
        gray_image = cvtColor(image, COLOR_BGR2GRAY)

        # face is a tuple with x, y, width and height; extract
        face_detected, face = face_cropper.get_face(gray_image)
        if face_detected:
            x, y, w, h = face
            
            # Retrieve an image of correct proportions and pass through emotion detection model

            # Display emotion and emoji; put rect around face
            pad = int(np.floor(w / 4))
            rectangle(image, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 4)

            fcimage = face_cropper.get_face_img_for_emotion(gray_image, face)

            now = time.time()
            # Save at 0.5 second intervals, and give a 10 second window to get ready before saving
            if now - prev > 0.5 and now - start > 5:
                # Save image cropped to face
                img_name = f"{counter}.png"
                path = os.path.join(os.getcwd(), "data", emoji, img_name)
                fullpath = path = os.path.join(os.getcwd(), "data_full", emoji, img_name)
                imwrite(path, fcimage)
                imwrite(fullpath, frame) # retain full screenshot for later
                print(f"Written image number {counter}")

                counter += 1
                prev = now

        key = waitKey(20)
        if key == 27: # exit on ESC
            vc.release()
            destroyWindow('Emojions')
            waitKey(1)

# perhaps actually irrelevant.
def mainfix():
    """
    I messed up with the image size from my initial data runs, should have been 48x48
    This is a method to convert the existing images back to 48x48 greyscale as needed
    """
    # classifications = ["devil", "flushed", "grin", "sad", "silly", "smiling_hearts", "sob"]
    classifications = ["devil"]

    cropper = FaceCropper()

    datacopy = os.path.join(os.getcwd(), "data_copy")
    data = os.path.join(os.getcwd(), "data")
    for cls in classifications:
        sourcepath = os.path.join(os.getcwd(), "data_copy", cls)
        destpath = os.path.join(os.getcwd(), "data", cls)
        for imgpath in os.listdir(sourcepath):
            src = os.path.join(sourcepath, imgpath)
            dst = os.path.join(destpath, imgpath)
            image = imread(src)
            yes, face = cropper.get_face(image)
            if yes:
                crop = cropper.get_face_img_for_emotion(image, face)
                imwrite(dst, crop)
            else:
                print(dst)

def train():
    model = EmojiModel()
    model.train()
    print("done ??")

if __name__=="__main__":
    main()
