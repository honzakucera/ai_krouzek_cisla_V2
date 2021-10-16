from PIL import Image
from tensorflow import keras
import random
from tensorflow.keras import layers
import tensorflow as tf
import os
import numpy as np


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def foto_to_vec():
    image = Image.open('dataset_numbers\\numbers\\0\\vzor0.jpg').convert('L')
    print(image.size)
    data = np.asarray(image).astype('float32')
    print(data)

#k demonstraci
#foto_to_vec()

input_dir = "dataset_numbers/numbers"

img_size = (24, 24)
num_classes = 10
batch_size = 1

tupple_lst = []

def load(dir):
    x,y,z = next(os.walk(dir))
    oper_lst = []

    y.sort(key=int)

    for i in y:
        b,bb,files = next(os.walk(x+'\\'+i))
        for j in files:
            image = Image.open(b+'\\'+j).convert('L')
            data = np.asarray(image)
            data = data.astype('float32')
            data_class = np.zeros(10).astype('float32')
            data_class[int(i)] = 1
            tupple_lst.append((data,data_class))
    random.shuffle(tupple_lst)

load('dataset_numbers\\numbers')

class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(tupple_lst)//self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        target_data = tupple_lst[i:i+self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size,dtype="float32")
        y = np.zeros((self.batch_size,) + (num_classes,), dtype="float32")
        for j, data in enumerate(target_data):
            x[j] = data[0]/255
            y[j] = data[1]
        return x,y



def get_model(img_size, num_classes):
    inputs = keras.Input(shape=(24,24))
    model = keras.Sequential(
        [   inputs,
            layers.Dense(8, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Flatten(),
            layers.Dense(num_classes, activation=tf.keras.activations.softmax)
        ]
    )
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()



# Build model
model = get_model(img_size, num_classes)
model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy)
model.summary()


val_samples = 5

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    1, img_size,
)



# Train the model, doing validation at the end of each epoch.
epochs = 40
model.fit(train_gen,epochs=epochs)


