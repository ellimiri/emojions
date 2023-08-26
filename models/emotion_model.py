from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import numpy as np

class EmotionModel:
    def __init__(self) -> None:
        self.emotion_dict = {
            0: "Angry", 
            1: "Disgusted", 
            2: "Fearful", 
            3: "Happy", 
            4: "Neutral", 
            5: "Sad", 
            6: "Surprised"
        }

        self.emoji_dict = {
            0: "😡",
            1: "😖",
            2: "😰",
            3: "😄",
            4: "😐",
            5: "😭",
            6: "😳"
        }
        self._init_atulapra()

    def _init_saranshbht(self):
        """
        Saransh Bhatia
        """
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.load_weights('models/model_saranshbht.h5')


    def _init_atulapra(self):
        """
        Creates an emotion detection model based on pre-trained weights sourced from 
        https://github.com/atulapra/Emotion-detection/ (weights in emotion_detection_model.h5)
        """
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.load_weights('models/model_atulapra.h5')

    def get_full_prediction(self, face_img):
        return self.model.predict(face_img)
    
    def get_prediction_index(self, face_img):
        """
        Returns the text of the emotion predicted from the given face image.
        """
        prediction = self.model.predict(face_img)
        argmax_index = int(np.argmax(prediction))
        return argmax_index
    
    def get_emotion_prediction(self, face_img):
        """
        Returns the text of the emotion predicted from the given face image.
        """
        idx = self.get_prediction_index(face_img)
        return self.emotion_dict[idx]
    
    def get_emoji_prediction(self, face_img):
        idx = self.get_prediction_index(face_img)
        return self.emoji_dict[idx]
    