from data_processing import build_train_validation_test_dataset
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    # Check if GPU is used
    print(tf.config.list_physical_devices('GPU'))

    vgg = VGG16(weights='imagenet', include_top=False,
                input_tensor=Input(shape=(224, 224, 3)))

    vgg.trainable = False

    flatten = vgg.output
    flatten = Flatten()(flatten)

    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(27, activation="sigmoid")(bboxHead)

    model = Model(inputs=vgg.input, outputs=bboxHead)

    optimizer = Adam()
    model.compile(loss="mse", optimizer=optimizer)

    train_dataset, validation_dataset, test_dataset = build_train_validation_test_dataset()

    train_dataset = train_dataset.batch(32)
    validation_dataset = validation_dataset.batch(32)

    num_epochs = 30

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=num_epochs, verbose=1)

    parent_dir = os.getcwd()
    model_dir = os.path.join(parent_dir, "models")
    model.save(model_dir + '/' + 'VGG16_froze_weights.h5')

    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    X = np.arange(1.0, num_epochs + 1, 1.)
    fig, axis = plt.subplots(1, 2)
    axis[0].plot(X, loss)
    axis[0].set_title("Training loss")
    axis[1].plot(X, val_loss)
    axis[1].set_title("Validation loss")

    plt.savefig(model_dir + '/' + "VGG16_froze_weights.png")

