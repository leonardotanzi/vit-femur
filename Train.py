import tensorflow_addons as tfa
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit
from utils import *


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print('TensorFlow Version ' + tf.__version__)

    seed_everything()

    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 40
    visualize_map = False
    load = True

    train_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Train\\'
    test_path = 'D:\\Drive\\PelvisDicom\\FinalDataset\\Dataset\\Test\\'

    classes_list = ["A1", "A2", "A3", "B1", "B2", "B3"]

    model_name = "..\\Models\\ViT-supervised\\7classes_l16"

    datagen_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0.15,
                                                              # rotation_range=5,
                                                              # width_shift_range=0.1,
                                                              # height_shift_range=0.1,
                                                              # brightness_range=(-5, 5),
                                                              # zoom_range=0.1
                                                              )
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_gen = datagen_aug.flow_from_directory(directory=train_path,
                                            subset="training",
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    valid_gen = datagen_aug.flow_from_directory(directory=train_path,
                                            subset='validation',
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    test_gen = datagen.flow_from_directory(directory=test_path,
                                            classes=classes_list,
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
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
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

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

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="D:\\7classes_l16-evenbiggerdense-nogenerator_{epoch:02d}.hdf5",
                                                      monitor="val_accuracy",
                                                      verbose=1,
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      mode="max",
                                                      period=1)

    callbacks = [earlystopping, checkpointer, reduce_lr] # WarmupExponentialDecay(lr_base=0.0002, decay=0, warmup_epochs=2)]

    if not load:
        # class_weights_train = compute_weights(train_path)

        model.fit(x=train_gen,
                  steps_per_epoch=STEP_SIZE_TRAIN,
                  validation_data=valid_gen,
                  validation_steps=STEP_SIZE_VALID,
                  epochs=EPOCHS,
                  # class_weight=class_weights_train,
                  callbacks=callbacks)

        model.save(model_name + ".hdf5")

    model.summary()

    print(model.evaluate_generator(test_gen))
    print(model.evaluate_generator(valid_gen))

    predicted_classes = np.argmax(model.predict_generator(test_gen, steps=test_gen.n // test_gen.batch_size + 1), axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    confusionmatrix_norm = confusionmatrix / confusionmatrix.astype(np.float).sum(axis=1)
    confusionmatrix_norm.round(decimals=2)
    print(confusionmatrix_norm)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix_norm, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(true_classes, predicted_classes))

