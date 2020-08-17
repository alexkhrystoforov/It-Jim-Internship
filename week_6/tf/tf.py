import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from sklearn.model_selection import KFold

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization, \
    Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, Nadam, Adadelta, Adagrad
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K


# pip install split-folders
import splitfolders


if os.path.isdir("dataset"):
    splitfolders.ratio("dataset", output="datasets", seed=1337, ratio=(.8, 0, .2), group_prefix=None)
    shutil.rmtree('dataset')
    shutil.rmtree('datasets/val')
else:
    print('you have already split dataset')

# delete outlier
if os.path.isdir('datasets/train/n02138441/n0213844100000172.jpg'):
    os.remove('datasets/train/n02138441/n0213844100000172.jpg')
else:
    print('you have already deleted outliers')


def load_datasets(train_folder, test_folder):

    def load_images_from_folder(folders):
        images = []
        labels = []
        for folder in os.listdir(folders):
            if folder == '.DS_Store':
                continue
            for filename in os.listdir(folders + folder):
                img = cv2.imread(os.path.join(folders + folder, filename))
                label = folder
                if img is not None:
                    images.append(img)
                    labels.append(label)

        return images, labels

    path = 'datasets/'

    train_X, train_Y = load_images_from_folder(path + train_folder)
    test_X, test_Y = load_images_from_folder(path + test_folder)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    return train_X, train_Y, test_X, test_Y


train_X, train_Y, test_X, test_Y = load_datasets('train/', 'test/')


def normal_and_reshape(x):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    x = x.reshape(x.shape[0], 84, 84, 3)

    return x


train_X = normal_and_reshape(train_X)
test_X = normal_and_reshape(test_X)

train_Y = pd.get_dummies(train_Y)
test_Y = pd.get_dummies(test_Y)

input_shape = train_X.shape[1:]

print("train shape: ", train_X.shape)
print("train one-hot Y shape: ", train_Y.shape)
print("test shape: ", train_X.shape)
print("test one-hot Y shape: ", test_Y.shape)

print('input_shape: ', input_shape)


def train_models():

    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(84)(x)
    x = Activation("relu")(x)
    x = Dense(168)(x)
    x = Activation("relu")(x)
    x = Dense(336)(x)
    x = Activation("relu")(x)
    x = Dense(672)(x)
    x = Activation("relu")(x)
    x = Dense(1344)(x)
    x = Activation("relu")(x)
    x = Dense(2688)(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(5376)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    predictions = Dense(16, activation='softmax')(x)

    # setting up the model
    nn_model = Model(inputs=inputs, outputs=predictions)

    # had an error, when build model below, so I rebuild model above

    # nn_model = Sequential([
    #     Flatten(),

    #     Dense(84, activation='relu', input_shape=input_shape),

    #     Dense(168, activation='relu'),

    #     Dense(336, activation='relu'),

    #     Dense(672, activation='relu'),

    #     Dense(1344, activation='relu'),

    #     Dense(2688, activation='relu'),

    #     Dense(5376),
    #     BatchNormalization(),
    #     Activation("relu"),
    #     Dropout(0.3),

    #     Dense(16, activation='softmax'),
    # ])

    cnn_model = Sequential([

        Conv2D(84, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(168, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(336, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(672, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(1344),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.5),

        Dense(16, activation='softmax')])

    def get_model(tr_x, tr_y, val_x, val_y, model):

        #     model.summary()
        opt = Adam(learning_rate=0.0001)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

        datagen = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.3,
            shear_range=0.3,
        )

        datagen.fit(tr_x)

        es = EarlyStopping(monitor='val_accuracy',
                           mode='max',
                           restore_best_weights=True,
                           verbose=3,
                           patience=10)

        mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max',
                             save_best_only=True, verbose=2, save_weights_only=True)

        history = model.fit(datagen.flow(tr_x, tr_y, batch_size=64), epochs=100, validation_data=(val_x, val_y),
                            callbacks=[es, mc])

        model.load_weights("best_model.h5")

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

        loss_and_metrics = model.evaluate(val_x, val_y, batch_size=64)

        acc_score = loss_and_metrics[1]

        print('Test loss:', loss_and_metrics[0])
        print('Test accuracy:', acc_score)

        return model, acc_score

    scores = []
    models = [cnn_model, nn_model]
    all_models = []
    model_counter = 1
    kfold = KFold(n_splits=5, shuffle=True, random_state=999)

    for model in models:
        for k_fold, (tr_index, val_index) in enumerate(kfold.split(train_X, train_Y)):
            print("-----------")
            print(f'Fold {k_fold + 1}/5')
            print("-----------")

            tr_x, val_x = train_X[tr_index], train_X[val_index]
            tr_y, val_y = train_Y.iloc[tr_index], train_Y.iloc[val_index]

            cur_model, acc_score = get_model(tr_x, tr_y, val_x, val_y, model)
            all_models.append(cur_model)

            scores.append(acc_score)

            cur_model.save(str(model_counter) + '_model.h5')

        model_counter += 1
        print("mean accuracy score is %f" % np.mean(scores))


def predict():

    f = open("metric.txt", "w+")
    f.write('First model is CNN \n')
    f.write('Second model is NN \n')
    for i in range(1, 3):#
        reconstructed_model = load_model(f"{i}_model.h5")
        scores = reconstructed_model.evaluate(test_X, test_Y, batch_size=64)

        print(f'Accuracy for {i} model is : ', scores[1])
        print(f'Loss for {i} model is : ', scores[0])

        f.write(f'Accuracy for {i} model on test data is : ' + str(scores[1]) + '\n')
        f.write(f'Loss for {i} model on test data is : ' + str(scores[0]) + '\n')

    f.close()


encoder_dict = {
    9: 'n03417042',
    15: 'n09256479',
    3: 'n02138441',
    5: 'n02950826',
    7: 'n02981792',
    2: 'n02114548',
    1: 'n02091244',
    0: 'n01855672',
    11: 'n03584254',
    13: 'n03773504',
    14: 'n03980874',
    10: 'n03535780',
    6: 'n02971356',
    8: 'n03075370',
    12: 'n03770439',
    4: 'n02174001'
}


def infer():
    image_path = 'datasets/train/n01855672/n0185567200000071.jpg'
    img = cv2.imread(image_path)
    x_min = img.min(axis=(1, 2), keepdims=True)
    x_max = img.max(axis=(1, 2), keepdims=True)

    img = (img - x_min) / (x_max - x_min)
    img = img.reshape(-1, 84, 84, 3)

    for i in range(1, 3):
        reconstructed_model = load_model(f"{i}_model.h5")
        if i == 1:
            pred_y = reconstructed_model.predict_classes(img, batch_size=64)
            print('predicted class by cnn is: ', encoder_dict.get(pred_y[0]))
        else:
            pred_y = reconstructed_model.predict(img, batch_size=64)
            print('predicted class by nn is: ', encoder_dict.get(int(np.argmax(pred_y, axis=1))))


if __name__ == '__main__':
    # train_models()
    # predict()
    infer()
