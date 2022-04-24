import tensorflow as tf
import os
import cv2
import numpy as np
import glob
import random
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy import stats
import xlrd


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
            if self.params['steps'] is None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        # se siamo nel warm up
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr, 0.01)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self, epoch, logs=None):
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


def compute_weights(input_folder):
    dictio = {"A1": 0, "A2": 1, "A3": 2, "B1": 3, "B2": 4, "B3": 5, "Unbroken": 6}
    files_per_class = []
    for folder in os.listdir(input_folder):
        if folder.startswith('.'):
            continue
        if folder in ["A", "B"]:
            continue
        if not os.path.isfile(folder):
            a = dictio.get(folder)
            files_per_class.insert(dictio.get(folder), (len(os.listdir(input_folder + '/' + folder))))
    total_files = sum(files_per_class)
    class_weights = {}
    for i in range(len(files_per_class)):
        class_weights[i] = 1 - (float(files_per_class[i]) / total_files)
    return class_weights


def from7to3classes(y):
    for i in range(len(y)):
        if y[i] == 1 or y[i] == 2:
            y[i] = 0
        elif y[i] == 3 or y[i] == 4 or y[i] == 5:
            y[i] = 1
        elif y[i] == 6:
            y[i] = 2
    return y


def from7to2classes(y):
    for i in range(len(y)):
        if y[i] == 1 or y[i] == 2 or y[i] == 3 or y[i] == 4 or y[i] == 5:
            y[i] = 0
        elif y[i] == 6:
            y[i] = 1
    return y


def build_dataset(train_path, classes_list):

    X = []
    Y = []

    for classes_name in classes_list:
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

    X = np.asarray(X)
    y = np.asarray(Y)

    np.savez_compressed("..\\NumpyData\\X_test_images.npz", X)
    np.savez_compressed("..\\NumpyData\\y_test_images.npz", y)


def k_fold(K):
    image_size = 224
    # categories = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    #path = "D:\\Drive\\PelvisDicom\\FinalDataset\\"
    categories = ["YES", "NO"]
    path = "D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\Dataset\\"

    input_folders = []
    for cat in categories:
        input_folders.append(path + cat)

    # create the root folder
    output_path = "D:\\Drive\\BeautyClassifier-POLI_MOLINETTE\\KCrossVal\\"
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
                # img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)  # convert to array
                new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
                X.append(new_array)  # add this to our training_data
                X_original.append(img_array)
                shapes.append(img_array.shape)
                names.append(img)

            except Exception as e:  # in the interest in keeping the output clean...
                pass

        # X = np.array(X).reshape(-1, image_size, image_size, 1)
        X = np.array(X).reshape(-1, image_size, image_size, 3)

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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def read_excel(loc):

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    sheet.cell_value(0, 0)

    correct = [0 for i in range(65)]
    for i in range(1, sheet.nrows):
        a = sheet.row_values(i)
        k = 0
        for j in range(7, sheet.ncols, 3):
            if a[j+1] == 1:
                correct[j-7-2*k] = a[j]
            k += 1
    correct[9] = "A1"

    sheet.cell_value(0, 0)

    total_spec = []
    for i in range(1, sheet.nrows):
        a = sheet.row_values(i)
        k = 0
        tot = 0
        for j in range(7, sheet.ncols, 3):
            x = a[j][0:1]
            if a[j][0:1] == correct[k][0:1]:
                tot += 1
            k += 1
        total_spec.append(tot)

    arr = np.uint8(np.asarray(total_spec) * 100 / 65)
    return arr

