import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import argparse
import numpy as np
import glob
import cv2
import os
import shutil
from CNN import *


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
#
# class1 = "Broken"
# class2 = "Unbroken"
#
# subclass1 = "A"
# subclass2 = "B"
# subclass3 = "Unbroken"
#
# label = "Unbroken"
#
# model_path = "C:\\Users\\d053175\\Desktop\\ViT\\ViT\\Models\\ViT-supervised\\selection\\Cascade\\"
# score_folder = "D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\{}".format(label)
# score_folder_A1A2A3 = "D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\A3"
# test_folder = ["D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\" + subclass1,
#                "D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\" + subclass2,
#                "D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\" + subclass3]
#
#
# output_path = "Cascade\\OutputBroUnbro\\"
# output_path_AB = "Cascade\\OutputAB\\"
#
# image_size = 299
#
# first_model = load_model(model_path + "2classesBroUnbro_ResNet-20.hdf5")
# second_model = load_model(model_path + "2classes_AB_ResNet-20.hdf5")
#
# i = 0
# j = 0
#
# for img_path in sorted(glob.glob(score_folder_A1A2A3 + "\\*.png"), key=os.path.getsize):
#
#     img = image.load_img(img_path, target_size=(image_size, image_size))
#     X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
#     original_name = img_path.split("/")[-1].split(".")[0]
#
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#
#     preds = first_model.predict(x)
#
#     class_idx = np.argmax(preds, axis=1)
#
#     if class_idx == 1:
#         print("Unbroken")
#         i += 1
#
#     elif class_idx == 0:
#         print("Broken")
#         j += 1
#         name_out = output_path + "{}".format(img_path.split("/")[-1])
#         cv2.imwrite(name_out, X_original)
#
# print("Unbroken {} - Broken {}".format(i, j))
#
# i = 0
# j = 0
#
# for img_path in sorted(glob.glob(output_path + "*.png"), key=os.path.getsize):
#
#     img = image.load_img(img_path, target_size=(image_size, image_size))
#     X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
#     original_name = img_path.split("/")[-1].split(".")[0]
#
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#
#     preds = second_model.predict(x)
#
#     class_idx = np.argmax(preds, axis=1)
#
#     if class_idx == 0:
#         print("A")
#         i += 1
#         # cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/APredicted/{}-Label{}-PredictedA.png".format(original_name, label), X_original)
#         name_out = output_path_AB + "{}".format(img_path.split("/")[-1])
#         cv2.imwrite(name_out, X_original)
#
#     elif class_idx == 1:
#         # print("B")
#         j += 1
#         # cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/BPredicted/{}-Label{}-PredictedB.png".format(original_name, label), X_original)
#
#
# print("A {} - B {}".format(i, j))
#
#
# i = 0
# j = 0
# k = 0
#
# classic_cascade = False
#
# if classic_cascade:
#
#     third_model = load_model(model_path + "Fold3_A1A2A3_notflipped-retrainAll-categorical-Inception-1569509422.model")
#
#     for img_path in sorted(glob.glob(output_path_AB + "*.png"), key=os.path.getsize):
#
#         img = image.load_img(img_path, target_size=(image_size, image_size))
#         X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
#
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#
#         preds = third_model.predict(x)
#
#         class_idx = np.argmax(preds, axis=1)
#
#         if class_idx == 0:
#             print("A1")
#             i += 1
#
#         elif class_idx == 1:
#             print("A2")
#             j += 1
#
#         elif class_idx == 2:
#             print("A3")
#             k += 1
# else:
#
#     third_model_A1A2 = load_model(model_path + "Fold1_A1_A2-binary-baselineInception-1569514982.model")
#     third_model_A1A3 = load_model(model_path + "Fold1_A1_A3-binary-baselineInception-1569535118.model")
#     third_model_A2A3 = load_model(model_path + "Fold3_A2_A3-binary-baselineInception-1569598028.model")
#     third_model = load_model(model_path + "Fold3_A1A2A3_notflipped-retrainAll-categorical-Inception-1569509422.model")
#
#     for img_path in sorted(glob.glob(output_path_AB + "*.png"), key=os.path.getsize):
#
#         img = image.load_img(img_path, target_size=(image_size, image_size))
#         X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
#
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#
#         predsA1A2 = third_model_A1A2.predict(x)  # 0 se A1, 1 se A2
#         predsA1A3 = third_model_A1A3.predict(x)  # 0 se A1, 1 se A3
#         predsA2A3 = third_model_A2A3.predict(x)  # 0 se A2, 1 se A3
#
#         preds = third_model.predict(x)
#
#         A1val = predsA1A2[0][0] + predsA1A3[0][0] + preds[0][0]
#         A2val = predsA1A2[0][1] + predsA2A3[0][0] + preds[0][1]
#         A3val = predsA1A3[0][1] + predsA2A3[0][1] + preds[0][2]
#
#         values = [[A1val, A2val, A3val]]
#
#         class_idx = np.argmax(values, axis=1)
#
#         if class_idx == 0:
#             print("A1")
#             i += 1
#
#         elif class_idx == 1:
#             print("A2")
#             j += 1
#
#         elif class_idx == 2:
#             print("A3")
#             k += 1
#
# print("A1 {} - A2 {} - A3 {}".format(i, j, k))
#
#
# shutil.rmtree(output_path)
# shutil.rmtree(output_path_AB)
# os.mkdir(output_path)
# os.mkdir(output_path_AB)