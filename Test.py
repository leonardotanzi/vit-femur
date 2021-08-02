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
import xlwt
from xlwt import Workbook
import xlrd


if __name__ == "__main__":

    seed_everything()

    visualize_map = False
    visualize_map_no_gen = False
    test_overall = True
    read_excel = False
    write_excel = False
    shuffle = False
    if visualize_map:
        shuffle = True

    IMAGE_SIZE = 224
    BATCH_SIZE = 16

    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\TestDoctors\\'
    model_name = "..\\Models\\ViT-supervised\\selection\\7classes_l16-evenbiggerdense-oversampling-08" #"..\\Models\\ViT-supervised\\7classes_l16-evenbiggerdense"

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0)

    test_gen = datagen.flow_from_directory(directory=test_path,
                                           classes=classes_list,
                                           batch_size=BATCH_SIZE,
                                           seed=1,
                                           color_mode='rgb',
                                           shuffle=shuffle,
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

    model.load_weights(model_name + ".hdf5")

    learning_rate = 1e-4

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])

    if visualize_map:
        x = test_gen.next()
        i = 0
        for i in range(test_gen.batch_size):
            image = x[0][i]
            label = x[1][i]
            label = classes_list[np.argmax(label)]
            image_map = image * 255
            image_map = np.uint8(image_map)

            prediction = model.predict(np.expand_dims(image, axis=0))
            label_pred = classes_list[np.argmax(prediction)]

            # image = cv2.imread(os.path.join(test_path, "A\\Pelvis00009_right_0.99.png"))
            # image = cv2.resize(image, (224, 224))
            # image *= 255
            attention_map = visualize.attention_map(model=model.get_layer("vit-l16"), image=image_map)

            # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Original', 400, 400)
            # cv2.namedWindow('Attention', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Attention', 400, 400)
            # cv2.imshow('Original', image_map)
            # print("Original {}, Predicted {}".format(label, label_pred))
            # cv2.imshow("Attention", attention_map)
            # cv2.waitKey()
            cv2.imwrite("..\\Map\\Original-{}-{}.png".format(label, i), image_map)
            cv2.imwrite("..\\Map\\Map-{}-{}-{}.png".format(label, label_pred, i), attention_map)

            i+=1

    if visualize_map_no_gen:

        X_dict_test = np.load("..\\NumpyData\\X_nogen_test.npz")
        X_test = X_dict_test['arr_0']
        y_dict_test = np.load("..\\NumpyData\\y_nogen_test.npz")
        y_test = y_dict_test['arr_0']

        i = 0
        for x, y in zip(X_test, y_test):

            label = classes_list[np.argmax(y)]
            image_map = x * 255
            image_map = np.uint8(image_map)

            prediction = model.predict(np.expand_dims(x, axis=0))
            label_pred = classes_list[np.argmax(prediction)]

            # image = cv2.imread(os.path.join(test_path, "A\\Pelvis00009_right_0.99.png"))
            # image = cv2.resize(image, (224, 224))
            # image *= 255
            attention_map = visualize.attention_map(model=model.get_layer("vit-l16"), image=image_map)

            # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Original', 400, 400)
            # cv2.namedWindow('Attention', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Attention', 400, 400)
            # cv2.imshow('Original', image_map)
            # print("Original {}, Predicted {}".format(label, label_pred))
            # cv2.imshow("Attention", attention_map)
            # cv2.waitKey()
            cv2.imwrite("..\\Map\\Original-{}-{}.png".format(label, i), image_map)
            cv2.imwrite("..\\Map\\Map-{}-{}-{}.png".format(label, label_pred, i), attention_map)
            i += 1

    if test_overall:

        print(model.evaluate_generator(test_gen))

        predicted_classes = np.argmax(model.predict_generator(test_gen, steps=test_gen.n // test_gen.batch_size + 1),
                                      axis=1)
        true_classes = test_gen.classes
        class_labels = list(test_gen.class_indices.keys())

        confusionmatrix = confusion_matrix(true_classes, predicted_classes)
        print(confusionmatrix)
        plt.figure(figsize=(16, 16))
        sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
        plt.show()

        # confusionmatrix_norm = confusionmatrix / confusionmatrix.astype(np.float).sum(axis=1)
        # confusionmatrix_norm.round(decimals=2)
        confusionmatrix_norm = np.around(confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis], decimals=2)

        print(confusionmatrix_norm)
        plt.figure(figsize=(16, 16))
        sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
        plt.show()

        print(classification_report(true_classes, predicted_classes))

    if write_excel:

        X_t = []
        Y_t = []
        file_names = []
        for c in classes_list:
            if os.path.isdir(os.path.join(test_path, c)):
                for file_name in glob.glob(os.path.join(test_path, c) + "//*.png"):
                    image = cv2.imread(file_name, cv2.COLOR_GRAY2RGB)

                    if len(image.shape) < 3:
                        image = np.stack((image,) * 3, axis=-1)
                    else:
                        print(image.shape)
                        print(file_name)

                    file_names.append(file_name.split("\\")[-1].split(".")[0] + "." + file_name.split("\\")[-1].split(".")[1])
                    image = cv2.resize(image, (224, 224))
                    X_t.append(image)
                    y_t = [0] * len(classes_list)
                    y_t[classes_list.index(c)] = 1
                    Y_t.append(y_t)

        c = list(zip(X_t, file_names, Y_t))

        random.shuffle(c)

        X_t, file_names, Y_t = zip(*c)
        # random.shuffle(X_t)
        X_test = np.asarray(X_t) / 255.0
        y_test = np.asarray(Y_t)

        predicted_classes = model.predict(X_test)
        predicted_classes_digits = np.argmax(predicted_classes, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        wb = Workbook()

        sheet = wb.add_sheet('ViT-prediction')

        sheet.write(0, 0, "Name")
        sheet.write(0, 1, "A1 (%)")
        sheet.write(0, 2, "A2 (%)")
        sheet.write(0, 3, "A3 (%)")
        sheet.write(0, 4, "B1 (%)")
        sheet.write(0, 5, "B2 (%)")
        sheet.write(0, 6, "B3 (%)")
        sheet.write(0, 7, "Unbroken (%)")
        sheet.write(0, 8, "Prediction")
        sheet.write(0, 9, "Real")

        num_row = 1

        for i in range(len(file_names)):
            sheet.write(num_row, 0, file_names[i])
            sheet.write(num_row, 1, str(np.uint8(predicted_classes[i][0] * 100)))
            sheet.write(num_row, 2, str(np.uint8(predicted_classes[i][1] * 100)))
            sheet.write(num_row, 3, str(np.uint8(predicted_classes[i][2] * 100)))
            sheet.write(num_row, 4, str(np.uint8(predicted_classes[i][3] * 100)))
            sheet.write(num_row, 5, str(np.uint8(predicted_classes[i][4] * 100)))
            sheet.write(num_row, 6, str(np.uint8(predicted_classes[i][5] * 100)))
            sheet.write(num_row, 7, str(np.uint8(predicted_classes[i][6] * 100)))
            sheet.write(num_row, 8, classes_list[np.uint8(predicted_classes_digits[i])])
            sheet.write(num_row, 9, classes_list[np.uint8(true_classes[i])])

            num_row += 1

        wb.save("ViT-prediction.xls")





