from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_processing import build_train_validation_test_dataset
import numpy as np
import matplotlib.pyplot as plt
import os


input_layer = Input(shape=(224, 224, 3))
x = Conv2D(128, (3, 3))(input_layer)
x = MaxPool2D()(x)
x = Conv2D(64, (3, 3))(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3))(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(27, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=x)

model.compile(optimizer=Adam(), loss="mse")

train_dataset, val_dataset, test_dataset = build_train_validation_test_dataset(size_of_dataset=0.1)
train_dataset = train_dataset.batch(8)
val_dataset = val_dataset.batch(8)

num_epochs = 10

history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

model.save("models/first_custom_model.h5")

parent_dir = os.getcwd()
model_dir = os.path.join(parent_dir, "models")

history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']

# build the train and validation loss plot
X = np.arange(1.0, num_epochs + 1, 1.)
fig, axis = plt.subplots(1, 2)
axis[0].plot(X, train_loss)
axis[0].set_title("Training loss")
axis[1].plot(X, val_loss)
axis[1].set_title("Validation loss")

# save the plot
plt.savefig(model_dir + '/' + "VGG16_unfroze_weights_15epochs.png")