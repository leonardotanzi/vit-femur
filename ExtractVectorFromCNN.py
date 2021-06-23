import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model, Model
from vit_keras import vit
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
import seaborn as sns
from utils import *

if __name__ == "__main__":

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 20

    model = load_model("7classes_cnn_1024.h5")

    layer_name = "dense"
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.15,
                                                              preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(directory=test_path,
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    out_intermediate = intermediate_layer_model.predict_generator(train_gen)
    true_value = train_gen.classes

    intermediate_layer_model.summary()

    np.savez_compressed("..\\NumpyData\\X_test_cnn.npz", out_intermediate)
    np.savez_compressed("..\\NumpyData\\y_test_cnn.npz", true_value)