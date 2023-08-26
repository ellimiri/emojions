from cv2 import flip, imshow, rectangle, waitKey, namedWindow, VideoCapture, destroyWindow, cvtColor, COLOR_BGR2GRAY, putText, FONT_HERSHEY_DUPLEX, LINE_AA
from utils.face_crop import FaceCropper
from models.emotion_model import EmotionModel
from utils.emoji_display import EmojiDisplay

from pyautogui import typewrite, hotkey

from pynput import keyboard
from pynput.keyboard import Events
from pyperclip import copy

def active(face_cropper, ed_model, emoji_display):
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
            face_image = face_cropper.get_face_img(gray_image, face)
            emotion = ed_model.get_emotion_prediction(face_image)
            emoji = ed_model.get_emoji_prediction(face_image)

            # Display emotion and emoji; put rect around face
            image = emoji_display.compose_images(image, emoji)
            putText(image, emotion, (x, y), FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, LINE_AA)
            rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

        key = waitKey(20)
        if key == 27: # exit on ESC
            vc.release()
            destroyWindow('Emojions')
            waitKey(1)
            return emoji

def main():
    face_cropper: FaceCropper = FaceCropper()
    ed_model = EmotionModel(which=1)
    emoji_display = EmojiDisplay()

    def activate_thingo():
        return active(face_cropper, ed_model, emoji_display)

    with keyboard.Events() as events:
        for event in events:
            if isinstance(event, Events.Release) and event.key == keyboard.Key.up:
                emoji = activate_thingo()
                # Go to the active window you were on before, and paste the emoji
                hotkey("command", "tab", interval=0.25)
                copy(emoji)
                hotkey("command", "v", interval=0.25)
            elif isinstance(event, Events.Release) and event.key == keyboard.Key.down:
                break

if __name__ == "__main__":
    main()
