import numpy as np
from time import time
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt


def acc_cluster(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    s = 0
    for i, j in zip(ind[0], ind[1]):
        s += w[i, j]
    return s * 1.0 / y_pred.size


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        # crea i pesi del livello, che sono n_clusters x 10 (encoded vector)
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), name='clustering', initializer='glorot_uniform')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


if __name__ == "__main__":

    pretrain_autoencoder = False
    train_cluster = True

    X_dict = np.load("X.npz")
    X = X_dict['arr_0']
    y_dict = np.load("y.npz")
    y = y_dict['arr_0']

    X_dict_test = np.load("X_test.npz")
    X_test = X_dict_test['arr_0']
    y_dict_test = np.load("y_test.npz")
    y_test = y_dict_test['arr_0']

    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    y_pred_kmeans = kmeans.fit_predict(X_test)
    print("Kmeans accuracy:{}".format(acc_cluster(y_test, y_pred_kmeans)))

    dims = [X.shape[-1], 500, 500, 2000, 10]
    # Generalization of Xavier inizialization
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = SGD(lr=0.01, momentum=0.9)
    pretrain_epochs = 300
    batch_size = 128

    autoencoder, encoder = autoencoder(dims, init=init)

    autoencoder.summary()

    autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

    if pretrain_autoencoder:
        autoencoder.fit(X, X, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        autoencoder.save("ae_weights.h5")
        encoder.save("e_weights.h5")
    else:
        autoencoder = load_model("ae_weights.h5")
        encoder = load_model("e_weights.h5")

    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    # inizializza i centri del cluster a quelli del kmeans.
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(X))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    if train_cluster:
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

                if ite == 0:
                    y_pred_last = np.copy(y_pred)

                if y is not None:
                    acc_v = np.round(acc_cluster(y, y_pred), 5)
                    nmi_v = np.round(nmi(y, y_pred), 5)
                    ari_v = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_v, nmi_v, ari_v), ' ; loss=', loss)

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[index * batch_size: min((index + 1) * batch_size, X.shape[0])]
            loss = model.train_on_batch(x=X[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0

        model.save("DEC_model_final.h5")

    else:
        model = load_model("DEC_model_final.h5")

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