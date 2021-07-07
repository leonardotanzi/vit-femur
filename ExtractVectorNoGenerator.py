from utils import *


if __name__ == "__main__":

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

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

    np.savez_compressed("..\\NumpyData\\X_nogen.npz", X)
    np.savez_compressed("..\\NumpyData\\y_nogen.npz", y)

    X = []
    Y = []
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
                X.append(image)
                y = [0] * len(classes_list)
                y[classes_list.index(c)] = 1
                Y.append(y)

    X_test = np.asarray(X) / 255.0
    y_test = np.asarray(Y)

    np.savez_compressed("..\\NumpyData\\X_nogen_test.npz", X_test)
    np.savez_compressed("..\\NumpyData\\y_nogen_test.npz", y_test)