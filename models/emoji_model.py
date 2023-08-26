import numpy as np
import PIL
import tensorflow
import pathlib

from keras import Model
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory

from models.emotion_model import EmotionModel

import os

class EmojiModel:
    def __init__(self):
        self.em = EmotionModel()
        base_model = self.em.model
        self.model = Sequential()
        # Copy over all layers in the base emotion recog model except for the last two
        for layer in base_model.layers:
            layer.trainable = False
            self.model.add(layer)
        # We want the last two models to relearn their weights
        for layer in self.model.layers[-2:]:
            layer.trainable = True
    
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
