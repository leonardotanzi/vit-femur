import tensorflow as tf
import os
import cv2
import numpy as np
import glob
import random
from sklearn.model_selection import KFold
from tqdm import tqdm


def k_fold(K=7):
    image_size = 224
    categories = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    path = "D:\\Drive\\PelvisDicom\\FinalDataset\\"

    input_folders = []
    for cat in categories:
        input_folders.append(path + cat)

    # create the root folder
    output_path = "D:\\Drive\\PelvisDicom\\KCrossVal\\"
    os.mkdir(output_path)

    # create folders for splitting the dataset
    for i in range(K):
        os.chdir(output_path)
        name_fold = "Fold{}".format(i + 1)
        os.mkdir(name_fold)
        os.chdir(output_path + "/" + name_fold)
        os.mkdir("Test")
        os.chdir(output_path + "/" + name_fold + "/Test")
        for cat in categories:
            os.mkdir(cat)
        os.chdir("..")
        os.mkdir("Train")
        os.chdir(output_path + "/" + name_fold + "/Train")
        for cat in categories:
            os.mkdir(cat)

    for enum, path in enumerate(input_folders):

        X = []
        X_original = []
        names = []
        shapes = []

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
                X.append(new_array)  # add this to our training_data
                X_original.append(img_array)
                shapes.append(img_array.shape)
                names.append(img)

            except Exception as e:  # in the interest in keeping the output clean...
                pass

        X = np.array(X).reshape(-1, image_size, image_size, 1)

        kf = KFold(n_splits=7, random_state=None, shuffle=True)
        nFold = 1
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            for i in train_index:
                cv2.imwrite(output_path + "/Fold{}/Train/{}/{}".format(nFold, categories[enum], names[i]),
                            X_original[i])
            for i in test_index:
                cv2.imwrite(output_path + "/Fold{}/Test/{}/{}".format(nFold, categories[enum], names[i]), X_original[i])
            nFold += 1


def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

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

    return image


def create_data(input_dir, img_height, img_width, classes_list):
    X = []
    Y = []

    # classes_list = []  # ["NO", "YES"]
    # for dir in os.listdir(input_dir):
    #     if dir.endswith(".ini"):
    #         continue
    #     else:
    #         classes_list.append(dir)

    print(classes_list)
    for c in classes_list:
        if os.path.isdir(os.path.join(input_dir, c)):
            for file_name in glob.glob(os.path.join(input_dir, c) + "//*.png"):
                image = cv2.imread(file_name, cv2.COLOR_GRAY2RGB)
                image = cv2.resize(image, (img_height, img_width))
                X.append(image)
                y = [0] * len(classes_list)
                y[classes_list.index(c)] = 1
                Y.append(y)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'