from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential, load_model, Model
from utils import *

if __name__ == "__main__":

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 20

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(directory=test_path,
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    true_value = train_gen.classes

    batch_index = 0

    i = 0
    while batch_index <= train_gen.batch_index:
        data = train_gen.next()
        if i == 0:
            data_array = data[0]
        else:
            data_array = np.concatenate((data_array, data[0]), axis=0)
        batch_index = batch_index + 1
        i += 1
    # now, data_array is the numeric data of whole images

    np.savez_compressed("..\\NumpyData\\X_test_convae.npz", data_array)
    np.savez_compressed("..\\NumpyData\\y_test_convae.npz", true_value)