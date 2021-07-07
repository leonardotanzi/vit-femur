import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit
from utils import *
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from imblearn.over_sampling import RandomOverSampler, SMOTE
import sklearn


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
    visualize_map = False
    load = False
    oversample = True

    model_name = "D:\\3classes_l16-evenbiggerdense-15"  # "..\\Models\\ViT-supervised\\7classes_l16-evenbiggerdense-nogen-oversample_13"

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'

    # classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    classes_list = ["A", "B", "Unbroken"]

    if len(classes_list) == 7:
        X_dict = np.load("..\\NumpyData\\X_nogen.npz")
        X = X_dict['arr_0']
        y_dict = np.load("..\\NumpyData\\y_nogen.npz")
        y = y_dict['arr_0']

        X_dict_test = np.load("..\\NumpyData\\X_nogen_test.npz")
        X_test = X_dict_test['arr_0']
        y_dict_test = np.load("..\\NumpyData\\y_nogen_test.npz")
        y_test = y_dict_test['arr_0']

    else:
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

                    image = cv2.resize(image, (224, 224))
                    X_t.append(image)
                    y_t = [0] * len(classes_list)
                    y_t[classes_list.index(c)] = 1
                    Y_t.append(y_t)

        X_test = np.asarray(X_t) / 255.0
        y_test = np.asarray(Y_t)

    X, X_valid, y, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.15, random_state=1)

    if oversample:
        ros = RandomOverSampler(random_state=0)
        data_oversampled = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        X_res, y = ros.fit_resample(data_oversampled, y)
        X = X_res.reshape(-1, 224, 224, 3)
        unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
        print(dict(zip(unique, counts)))


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
                  loss="categorical_crossentropy", #tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])

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

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="D:\\3classes_l16-evenbiggerdense-oversampling-{epoch:02d}.hdf5",
                                                      monitor="val_accuracy",
                                                      verbose=1,
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      mode="max",
                                                      period=1)

    callbacks = [earlystopping, checkpointer, reduce_lr] # WarmupExponentialDecay(lr_base=0.0002, decay=0, warmup_epochs=2)]

    if not load:

        model.fit(x=X,
                  y=y,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(X_valid, y_valid),
                  # class_weight=class_weights_train,
                  callbacks=callbacks)

        model.save(model_name + ".hdf5")

    model.summary()

    print(model.evaluate(X_test, y_test, batch_size=BATCH_SIZE))

    predicted_classes = np.argmax(model.predict(X_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)
    # class_labels = list(test_gen.class_indices.keys())

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

    # CNN