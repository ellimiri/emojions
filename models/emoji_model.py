import numpy as np
import PIL
import tensorflow
import pathlib

from keras.applications import InceptionV3
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.utils import image_dataset_from_directory

from models.emotion_model import EmotionModel

import os

class EmojiModel:
    def __init__(self):
        self.emoji_dict = {
            "😳",
            "😄",
            "😐"
            "🙁",
            "😜",
            "🥰"
        }

        self.base = InceptionV3(include_top=False, input_shape=(256, 256, 3), classes=7, pooling="avg")
        self.model = Sequential()
        # Copy over all layers in the base emotion recog model except for the last two
        for layer in self.base.layers:
            layer.trainable = False
        self.model.add(self.base)
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))

    def train(self):
        dir = os.path.join(os.getcwd(), "data")
        # create image dataset
        training_dataset = image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset="training",
            seed=22)
        validation_dataset = image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset="validation",
            seed=22)
        
        epochs = 10
        lr = 0.001
        
        self.model.compile(optimizer=Adam(lr=lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        history = self.model.fit(training_dataset, validation_data=validation_dataset, epochs=epochs)

        self.model.save_weights('emoji.h5')
    
    def load(self):
        self.model.load_weights('emoji.h5')
    
    def get_prediction_index(self, face_img):
        prediction = self.model.predict(face_img)
        argmax_index = int(np.argmax(prediction))
        return argmax_index

    def get_emoji_prediction(self, face_img):
        idx = self.get_prediction_index(face_img)
        return self.emoji_dict[idx]


