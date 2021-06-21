import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from utils import *


class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = L.Dense(units = projection_dim)
        self.position_embedding = L.Embedding(
            input_dim = num_patches, output_dim = projection_dim
        )

    def call(self, patch):
        positions = tf.range(start = 0, limit = self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = L.Dense(units, activation=tf.nn.gelu)(x)
        x = L.Dropout(dropout_rate)(x)
    return x


def vision_transformer():
    inputs = L.Input(shape=(image_size, image_size, 3))

    # Create patches.
    patches = Patches(patch_size)(inputs)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = L.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = L.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = L.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = L.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = L.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = L.Flatten()(representation)
    representation = L.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # Classify outputs.
    logits = L.Dense(n_classes)(features)

    # Create the model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    return model


if __name__=="__main__":

    print('TensorFlow Version ' + tf.__version__)
    seed_everything()
    warnings.filterwarnings('ignore')

    classes = {0: "A1",
               1: "A2",
               2: "A3",
               3: "B1",
               4: "B2",
               5: "B3",
               6: "Unbroken"
               }

    classes_list = ["A", "B", "Unbroken"]
    # classes_list = ["A1", "A2", "A3", "B1", "B2", "B3", "Unbroken"]

    image_size = 224
    batch_size = 32
    n_classes = len(classes_list)

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\'

    # X, Y = create_data(train_path, image_size, image_size, classes_list)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)

    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
    #                                                           samplewise_std_normalization=True,
    #                                                           validation_split=0.2,
    #                                                           preprocessing_function=data_augment)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2,)

    train_gen = datagen.flow_from_directory(directory=train_path,
                                            subset="training",
                                            classes=classes_list,
                                            batch_size=batch_size,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(image_size, image_size))

    valid_gen = datagen.flow_from_directory(directory=train_path,
                                            subset='validation',
                                            classes=classes_list,
                                            batch_size=batch_size,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(image_size, image_size))

    # test_gen = datagen.flow_from_dataframe(dataframe = df_test,
    #                                        x_col = 'image_path',
    #                                        y_col = None,
    #                                        batch_size = batch_size,
    #                                        seed = 1,
    #                                        color_mode = 'rgb',
    #                                        shuffle = False,
    #                                        class_mode = None,
    #                                        target_size = (image_size, image_size))


    # images = [train_gen[0][0][i] for i in range(16)]
    # fig, axes = plt.subplots(3, 5, figsize=(10, 10))
    #
    # axes = axes.flatten()
    #
    # for img, ax in zip(images, axes):
    #     ax.imshow(img.reshape(image_size, image_size, 1))
    #     ax.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 30

    patch_size = 7  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [56, 28]

    # plt.figure(figsize=(4, 4))
    #
    # x = train_gen.next()
    # image = x[0][0]
    #
    # plt.imshow(image.astype('uint8'))
    # plt.axis('off')
    #
    # resized_image = tf.image.resize(
    #     tf.convert_to_tensor([image]), size=(image_size, image_size)
    # )
    #
    # patches = Patches(patch_size)(resized_image)
    # print(f'Image size: {image_size} X {image_size}')
    # print(f'Patch size: {patch_size} X {patch_size}')
    # print(f'Patches per image: {patches.shape[1]}')
    # print(f'Elements per patch: {patches.shape[-1]}')
    #
    # n = int(np.sqrt(patches.shape[1]))
    # plt.figure(figsize=(4, 4))
    #
    # for i, patch in enumerate(patches[0]):
    #     ax = plt.subplot(n, n, i + 1)
    #     patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    #     plt.imshow(patch_img.numpy().astype('uint8'))
    #     plt.axis('off')

    decay_steps = train_gen.n // train_gen.batch_size
    initial_learning_rate = learning_rate

    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = vision_transformer()
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=10,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./model.hdf5',
                                                      monitor='val_accuracy',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='max')

    callbacks = [earlystopping, lr_scheduler, checkpointer]

    model.fit(x=train_gen,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=valid_gen,
              validation_steps=STEP_SIZE_VALID,
              epochs=num_epochs,
              callbacks=callbacks)

    print('Training results')
    model.evaluate(train_gen)

    print('Validation results')
    model.evaluate(valid_gen)