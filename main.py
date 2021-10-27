import os
import random

import numpy as np
from PIL import Image
from tensorflow import keras


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Převod fotky na vektor, na ukázku
# def photo_to_vector():
#     image = Image.open('dataset_numbers\\numbers\\0\\vzor0.jpg').convert('L')
#     print(image.size)
#     data = np.asarray(image).astype('float32')
#     print(data)


# photo_to_vector()

# Globální parametry
input_dir = os.path.join('dataset_numbers', 'numbers')
test_dir = os.path.join('dataset_numbers', 'test')

img_size = (24, 24)
num_classes = 10
batch_size = 1

tuple_lst = []


# Načíst tréningová data
def load_dataset(root):
    tuple_lst = []
    dirs = next(os.walk(root))[1]
    dirs.sort(key=int)

    for dirname in dirs:
        data_dir = os.path.join(root, dirname)
        files = next(os.walk(data_dir))[2]

        for file in files:
            image = Image.open(os.path.join(data_dir, file)).convert('L')
            data = np.asarray(image)
            data = data.astype('float32')
            data_class = np.zeros(10).astype('float32')
            data_class[int(dirname)] = 1
            tuple_lst.append((data, data_class))

    random.shuffle(tuple_lst)
    return tuple_lst


data_with_labels = load_dataset(input_dir)


# Pomocník pro iteraci nad daty pro náš model
class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, data_with_labels, batch_size, img_size, num_classes):
        self.data_with_labels = data_with_labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data_with_labels)//self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        target_data = self.data_with_labels[i:i+self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        y = np.zeros((self.batch_size,) + (self.num_classes,), dtype="float32")
        for j, data in enumerate(target_data):
            x[j] = data[0]/255
            y[j] = data[1]
        return x, y


# Definice jednotlivých vrstev modelu
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size)
    model = keras.Sequential(
        [
            inputs,
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation=keras.activations.softmax)
        ]
    )
    return model


# Uvolnit RAM pro případ že spustíme definici modelu vícekrát
keras.backend.clear_session()


# Sestavit model
model = get_model(img_size, num_classes)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
model.summary()


val_samples = 5

# Instantiate data Sequences for each split
train_gen = OxfordPets(data_with_labels, batch_size, img_size, num_classes)


# Vytrénovat model (validuje se na konci každé epochy)
epochs = 40
model.fit(train_gen, epochs=epochs)
