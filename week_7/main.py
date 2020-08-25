import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")

x_train_rotated = copy.copy(x_train)
x_test_rotated = copy.copy(x_test)


def rotate_dataset(x):
    for i in range(len(x)):
        x[i] = cv2.rotate(x[i], cv2.ROTATE_90_CLOCKWISE)
    return x


x_train_rotated = rotate_dataset(x_train_rotated)
x_test_rotated = rotate_dataset(x_test_rotated)


# plt.imshow(x_test[0])
# plt.show()

# plt.imshow(x_test_rotated[0])
# plt.show()


def normal_and_reshape(x):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    x = x.reshape(x.shape[0], 28, 28, 1)

    return x


x_train = normal_and_reshape(x_train)
x_test = normal_and_reshape(x_test)

x_train_rotated = normal_and_reshape(x_train_rotated)
x_test_rotated = normal_and_reshape(x_test_rotated)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

input_shape = x_train.shape[1:]

print("train shape: ", x_train.shape)
print("train one-hot Y shape: ", y_train.shape)
print("test shape: ", x_test.shape)
print("test one-hot Y shape: ", y_test.shape)

print('input_shape: ', input_shape)

cnn_model = Sequential([

    Conv2D(28, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),

    Conv2D(56, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(112, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(224, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(448),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.5),

    Dense(10, activation='softmax')])


def get_history(history):
    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


def train_model(tr_x, tr_y, val_x, val_y, model, mc=True, load_weights=False):
    opt = Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    if load_weights:
        model.load_weights("best_weights.h5")

    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        shear_range=0.3,
    )

    datagen.fit(tr_x)

    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       restore_best_weights=True,
                       verbose=3,
                       patience=10)
    if mc:
        mc = ModelCheckpoint('best_weights.h5', monitor='val_accuracy', mode='max',
                             save_best_only=True, verbose=3, save_weights_only=True)

        history = model.fit(datagen.flow(tr_x, tr_y, batch_size=256), epochs=5, validation_data=(val_x, val_y),
                            callbacks=[es, mc])
        model.load_weights("best_weights.h5")

    else:
        history = model.fit(datagen.flow(tr_x, tr_y, batch_size=256), epochs=5, validation_data=(val_x, val_y),
                            callbacks=[es])

    get_history(history)

    loss_and_metrics = model.evaluate(val_x, val_y, batch_size=256)

    acc_score = loss_and_metrics[1]

    print('Test loss:', loss_and_metrics[0])
    print('Test accuracy:', acc_score)

    return model, acc_score


def train():
    # task 1-2
    print("-----------")
    print('task 1-2....')
    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=999)

    for k_fold, (tr_index, val_index) in enumerate(kfold.split(x_train_rotated, y_train)):
        print("-----------")
        print(f'Fold {k_fold + 1}/5')
        print("-----------")

        tr_x, val_x = x_train_rotated[tr_index], x_train_rotated[val_index]
        tr_y, val_y = y_train.iloc[tr_index], y_train.iloc[val_index]

        rot_model, acc_score = train_model(tr_x, tr_y, val_x, val_y, cnn_model)

        scores.append(acc_score)

        rot_model.save('rotated_model.h5')

    print("mean accuracy score is %f" % np.mean(scores))

    # task 3-4
    print("-----------")
    print('task 3-4....')
    print("-----------")
    reconstructed_model = load_model("rotated_model.h5")
    # Freeze all the layers, except the last
    for layer in reconstructed_model.layers[:-1]:
        layer.trainable = False

    for layer in reconstructed_model.layers:
        print(layer, layer.trainable)

    new_model = Sequential()
    new_model.add(reconstructed_model)
    new_model.summary()

    tr_x, val_x, tr_y, val_y = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    new_model, _ = train_model(tr_x, tr_y, val_x, val_y, new_model, mc=False)

    new_model.save('retrained_CNN_a_model.h5')

    # task 5
    print("-----------")
    print('task 5....')
    print("-----------")

    tr_x, val_x, tr_y, val_y = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    new_model, _ = train_model(tr_x, tr_y, val_x, val_y, cnn_model, mc=False, load_weights=True)

    new_model.save('retrained_CNN_c_model.h5')


def save_scores():
    f = open("results.txt", "w+")

    reconstructed_model = load_model("rotated_model.h5")

    scores = reconstructed_model.evaluate(x_test, y_test, batch_size=256)
    f.write('Rotated model accuracy on normal test is :' + str(scores[1]) + '\n')
    f.write('Rotated model loss on normal test is ' + str(scores[0]) + '\n')

    scores = reconstructed_model.evaluate(x_test_rotated, y_test, batch_size=256)
    f.write('Rotated model accuracy on rotated test is :' + str(scores[1]) + '\n')
    f.write('Rotated model loss on rotated test is ' + str(scores[0]) + '\n')

    reconstructed_model = load_model("retrained_CNN_a_model.h5")

    scores = reconstructed_model.evaluate(x_test, y_test, batch_size=256)
    f.write('retrained CNN a):  accuracy on normal test is :' + str(scores[1]) + '\n')
    f.write('retrained CNN a):  loss on normal test is ' + str(scores[0]) + '\n')

    reconstructed_model = load_model("retrained_CNN_c_model.h5")

    scores = reconstructed_model.evaluate(x_test, y_test, batch_size=256)
    f.write('retrained CNN c):  accuracy on normal test is :' + str(scores[1]) + '\n')
    f.write('retrained CNN c):  loss on normal test is ' + str(scores[0]) + '\n')

    f.close()


if '__name__' == '__main__':
    # train()
    save_scores()
