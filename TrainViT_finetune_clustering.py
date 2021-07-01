import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit
from vit_keras import visualize
from utils import *
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from Clustering import ClusteringLayer, target_distribution, acc_cluster
from sklearn.cluster import KMeans
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from tensorflow.keras.models import Model

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


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print('TensorFlow Version ' + tf.__version__)

    seed_everything()

    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 40
    visualize_map = False
    load = False
    n_clusters = 7

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]
    # classes_list = ["A", "B", "Unbroken"]

    model_name = "..\\Models\\ViT-supervised\\7classes-unsupervised"

    X_dict = np.load("..\\NumpyData\\X_convae.npz")
    X = X_dict['arr_0']
    y_dict = np.load("..\\NumpyData\\y_convae.npz")
    y = y_dict['arr_0']

    X_dict_test = np.load("..\\NumpyData\\X_test_convae.npz")
    X_test = X_dict_test['arr_0']
    y_dict_test = np.load("..\\NumpyData\\y_test_convae.npz")
    y_test = y_dict_test['arr_0']

    vit_model = vit.vit_l16(
            image_size=IMAGE_SIZE,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=len(classes_list))

    encoder = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
            ],
        name='vision_transformer')

    if load:
        encoder.load_weights(model_name + ".hdf5")

    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.summary()

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)
    y_pred = kmeans.fit_predict(model.predict(X))

    y_pred_last = np.copy(y_pred)

    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    loss = 0
    index = 0
    maxiter = 8000
    update_interval = 140
    index_array = np.arange(X.shape[0])
    tol = 0.001  # tolerance threshold to stop training

    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(X, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)

            # if ite == 0:
            #     y_pred_last = np.copy(y_pred)

            if y is not None:
                acc_v = np.round(acc_cluster(y, y_pred), 5)
                nmi_v = np.round(nmi(y, y_pred), 5)
                ari_v = np.round(ari(y, y_pred), 5)
                loss = np.round(loss, 10)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_v, nmi_v, ari_v), ' ; loss=', loss)

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * BATCH_SIZE: min((index + 1) * BATCH_SIZE, X.shape[0])]
        loss = model.train_on_batch(x=X[idx], y=p[idx])
        index = index + 1 if (index + 1) * BATCH_SIZE <= X.shape[0] else 0

    # model.save("..\\Models\\ViT-unsupervised\\DEC_model_final.h5")

    # Eval.
    q = model.predict(X_test, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y is not None:
        acc_v = np.round(acc_cluster(y_test, y_pred), 5)
        nmi_v = np.round(nmi(y_test, y_pred), 5)
        ari_v = np.round(ari(y_test, y_pred), 5)
        loss = np.round(loss, 5)
        print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc_v, nmi_v, ari_v), ' ; loss=', loss)

    sns.set(font_scale=3)
    confusion_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    # Here you can quickly match the clustering assignment by hand, e.g., cluster 1 matches with true label 7 or handwritten digit "7" and vise visa.
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()
    #
    # learning_rate = 1e-4
    #
    # optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    #
    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
    #               metrics=['accuracy'])
    #
    # STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    # STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
    #
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
    #                                                  factor=0.2,
    #                                                  patience=4,
    #                                                  verbose=1,
    #                                                  min_delta=1e-4,
    #                                                  min_lr=1e-6,
    #                                                  mode='max')
    #
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
    #                                                  min_delta=1e-4,
    #                                                  patience=20,
    #                                                  mode='max',
    #                                                  restore_best_weights=True,
    #                                                  verbose=1)
    #
    # checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="{}_best.hdf5".format(model_name),
    #                                                   monitor="val_accuracy",
    #                                                   verbose=1,
    #                                                   save_best_only=True,
    #                                                   save_weights_only=True,
    #                                                   mode="max")
    #
    # callbacks = [earlystopping, checkpointer, reduce_lr] # WarmupExponentialDecay(lr_base=0.0002, decay=0, warmup_epochs=2)]
    #
    # if not load:
    #     model.fit(x=train_gen,
    #               steps_per_epoch=STEP_SIZE_TRAIN,
    #               validation_data=valid_gen,
    #               validation_steps=STEP_SIZE_VALID,
    #               epochs=EPOCHS,
    #               callbacks=callbacks)
    #
    #     model.save(model_name + ".hdf5")
    #
    # model.summary()
    #
    # print(model.evaluate_generator(test_gen))
    # print(model.evaluate_generator(valid_gen))
    #
    # predicted_classes = np.argmax(model.predict_generator(test_gen, steps=test_gen.n // test_gen.batch_size + 1), axis=1)
    # true_classes = test_gen.classes
    # class_labels = list(test_gen.class_indices.keys())
    #
    # confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    # print(confusionmatrix)
    # plt.figure(figsize=(16, 16))
    # sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    # plt.show()
    #
    # print(classification_report(true_classes, predicted_classes))
    #
    # # CNN
