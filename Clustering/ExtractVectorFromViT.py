from tensorflow.keras.models import load_model, Model
from vit_keras import vit
import tensorflow as tf
import cv2
import numpy as np
import os
import glob


if __name__ == "__main__":

    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20

    one_image = False

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    train_path = "D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\"

    vit_model = vit.vit_l16(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=len(classes_list))

    vit_model.summary()

    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=tf.nn.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(classes_list), 'softmax')
    ],
        name='vision_transformer')

    model.load_weights("..\\Models\\ViT-supervised\\selection\\7classes_l16-evenbiggerdense-oversampling-08.hdf5")

    model.summary()

    layer_name = "flatten"
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    X = []
    Y = []

    for c in classes_list:
        if os.path.isdir(os.path.join(train_path, c)):
            for file_name in glob.glob(os.path.join(train_path, c) + "//*.png"):
                image = cv2.imread(file_name, cv2.COLOR_GRAY2RGB)

                if len(image.shape) < 3:
                    image = np.stack((image,) * 3, axis=-1)
                else:
                    print(image.shape)
                    print(file_name)

                image = cv2.resize(image, (224, 224))
                X.append(image)
                y = [0] * len(classes_list)
                y[classes_list.index(c)] = 1
                Y.append(y)

    X = np.asarray(X) / 255.0
    y = np.asarray(Y)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0.0)
    train_gen = datagen.flow_from_directory(directory=train_path,
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    out_intermediate = intermediate_layer_model.predict(X)
    true_value = train_gen.classes

    intermediate_layer_model.summary()

    np.savez_compressed("..\\NumpyData\\X_final_test.npz", out_intermediate)
    np.savez_compressed("..\\NumpyData\\Y_final_test.npz", true_value)

    if one_image:
        test_img = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\test\\A\\Pelvis00020_left_0.99.png'

        image = cv2.imread(test_img, cv2.COLOR_BGR2RGB)
        stacked_img = np.stack((image,) * 3, axis=-1)

        image = cv2.resize(stacked_img, (224, 224))
        image = image.astype("float32")
        image /= 255.0
        X = np.expand_dims(image, axis=0)

        # X /= 255.0

        inter_output = intermediate_layer_model.predict(X)
