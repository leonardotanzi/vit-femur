from keras.applications.inception_v3 import preprocess_input
import numpy as np
import glob
import cv2
import os
from TrainCNN import *


def build_model(n_class):
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3), weights="imagenet", pooling="avg")
    out = Dense(4096, activation="relu")(model.output)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(n_class)(out)
    out = Activation("softmax")(out)

    return Model(inputs=model.input,
                  outputs=out)


if __name__ == "__main__":

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'
    model_path = "C:\\Users\\d053175\\Desktop\\ViT\\ViT\\Models\\ViT-supervised\\selection\\Cascade\\"

    X_t = []
    Y_t = []
    for c in classes_list:
        if os.path.isdir(os.path.join(test_path, c)):
            for file_name in glob.glob(os.path.join(test_path, c) + "//*.png"):
                image = cv2.imread(file_name, cv2.COLOR_GRAY2RGB)

                if len(image.shape) < 3:
                    image = np.stack((image,) * 3, axis=-1)
                else:
                    print(image.shape)
                    print(file_name)

                image = cv2.resize(image, (299, 299))
                X_t.append(image)
                y_t = [0] * len(classes_list)
                y_t[classes_list.index(c)] = 1
                Y_t.append(y_t)

    X_test = preprocess_input(np.asarray(X_t))
    y_test = np.asarray(Y_t)

    BroUnbro_model = build_model(2)
    BroUnbro_model.load_weights(model_path + "2classesBroUnbro_ResNet-20.hdf5")
    AB_model = build_model(2)
    AB_model.load_weights(model_path + "2classes_AB_ResNet-20.hdf5")
    A_model = build_model(3)
    A_model.load_weights(model_path + "3classesA_ResNet-16.hdf5")
    B_model = build_model(3)
    B_model.load_weights(model_path + "3classesB_ResNet-20.hdf5")

    precise_pred = []
    for x, y in zip(X_test, y_test):
        x = np.expand_dims(x, axis=0)
        pred = BroUnbro_model.predict(x)
        first_pred = np.argmax(pred, axis=1)

        if first_pred == 0:
            pred = AB_model.predict(x)
            second_pred = np.argmax(pred, axis=1)
            if second_pred == 0:
                # Qua siamo nelle A
                pred = A_model.predict(x)
                third_pred = np.argmax(pred, axis=1)
                if third_pred == 0:
                    precise_pred.append(classes_list.index("A1"))
                elif third_pred == 1:
                    precise_pred.append(classes_list.index("A2"))
                elif third_pred == 2:
                    precise_pred.append(classes_list.index("A3"))
            elif second_pred == 1:
                # Qua nelle B
                pred = B_model.predict(x)
                third_pred = np.argmax(pred, axis=1)
                if third_pred == 0:
                    precise_pred.append(classes_list.index("B1"))
                elif third_pred == 1:
                    precise_pred.append(classes_list.index("B2"))
                elif third_pred == 2:
                    precise_pred.append(classes_list.index("B3"))
        else:
            precise_pred.append(classes_list.index("Unbroken"))

    true_classes = np.argmax(y_test, axis=1)
    print(y)
    print(precise_pred)

    predicted_classes = np.asarray(precise_pred)

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    # confusionmatrix_norm = confusionmatrix / confusionmatrix.astype(np.float).sum(axis=1)
    # confusionmatrix_norm.round(decimals=2)
    confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis],
                                     decimals=2)

    print(confusionmatrix_norm)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))