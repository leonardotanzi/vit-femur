import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit
from vit_keras import visualize
from utils import *
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


class WarmupExponentialDecay(Callback):

    def __init__(self, lr_base=1e4, lr_min=1e6, decay=0.00002, warmup_epochs=2):
        self.num_passed_batchs = 0   #One counter
        self.warmup_epochs = warmup_epochs
        self.lr = lr_base #learning_rate_base
        self.lr_min = lr_min #Minimum initial learning rate, this code has not yet been implemented
        self.decay = decay  #Exponential decay rate
        self.steps_per_epoch = 0 #Also a counter

    def on_batch_begin(self, batch, logs=None):
        # params are some parameters that the model automatically passes to Callback
        if self.steps_per_epoch == 0:
            #Prevent changes when running the verification set
            if self.params['steps'] is None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        # se siamo nel warm up
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr, 0.01)
            #self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self, epoch, logs=None):
    #Used to output the learning rate, can be deleted
        print("learning_rate: {:.9f}".format(K.get_value(self.model.optimizer.lr)))

def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)

    return image


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print('TensorFlow Version ' + tf.__version__)

    seed_everything()

    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 40
    visualize_map = True
    load = True

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    # classes_list = ["A", "B", "Unbroken"]

    model_name = "..\\Models\\ViT-supervised\\7classes_l16-evenbiggerdense"

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0.15)

    train_gen = datagen.flow_from_directory(directory=train_path,
                                            subset="training",
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    valid_gen = datagen.flow_from_directory(directory=train_path,
                                            subset='validation',
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    test_gen = datagen.flow_from_directory(directory=test_path,
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    vit_model = vit.vit_l16(
            image_size=IMAGE_SIZE,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=len(classes_list))

    model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dense(1096, activation=tf.nn.gelu),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dense(len(classes_list), 'softmax')
            tf.keras.layers.Dense(4096, activation=tf.nn.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(classes_list), 'softmax')
            ],
        name='vision_transformer')

    if load:
        model.load_weights(model_name + ".hdf5")

    learning_rate = 1e-4

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.2,
                                                     patience=4,
                                                     verbose=1,
                                                     min_delta=1e-4,
                                                     min_lr=1e-6,
                                                     mode='max')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=20,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="./best_{}.hdf5".format(model_name),
                                                      monitor="val_accuracy",
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode="max")

    callbacks = [earlystopping, checkpointer, reduce_lr] # WarmupExponentialDecay(lr_base=0.0002, decay=0, warmup_epochs=2)]

    if not load:
        model.fit(x=train_gen,
                  steps_per_epoch=STEP_SIZE_TRAIN,
                  validation_data=valid_gen,
                  validation_steps=STEP_SIZE_VALID,
                  epochs=EPOCHS,
                  callbacks=callbacks)

        model.save(model_name + ".hdf5")

    model.summary()

    print(model.evaluate_generator(test_gen))
    print(model.evaluate_generator(valid_gen))

    predicted_classes = np.argmax(model.predict_generator(test_gen, steps=test_gen.n // test_gen.batch_size + 1), axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))

    # CNN
