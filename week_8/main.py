import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

from week_8.utils import *
from week_8.enet import *

DATA_PATH = 'foosball_dataset/'

learning_rate = 1e-4
loss = 'binary_crossentropy'
epochs = 3
batch_size = 100

dataset = Dataset(DATA_PATH)


tiles = dataset.split_image_to_tiles(dataset.all_pathes)

optimizer = Adam(lr=learning_rate)

# Build model

builder = ENet(n_classes=2, input_height=256, input_width=256)
model = builder.build_model()
model.summary()

# Dataset generators

train_generator = dataset.generator('train')
valid_generator = dataset.generator('val')

# Compile model

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train Model

model.fit_generator(generator=train_generator, epochs=epochs,
                    validation_data=valid_generator, validation_steps=30)
