import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations


def normalize_image(image):
    image = image / 255
    return image


class Dataset():
    def __init__(self, data_path):

        self.train_path = os.path.join(data_path + 'train_set')
        self.train_mask_path = os.path.join(data_path + 'train_set_mask')
        self.val_path = os.path.join(data_path + 'test_set')
        self.val_mask_path = os.path.join(data_path + 'test_set_mask')
        self.test_path = os.path.join(data_path + 'val_set')
        self.test_mask_path = os.path.join(data_path + 'val_set_mask')
        self.all_pathes = [self.train_path, self.train_mask_path, self.val_path, self.val_mask_path,
                      self.test_path, self.test_mask_path]
        self.augmentation = dict(rescale=1. / 255, horizontal_flip=True)
        self.tile_shape = (256, 256, 3)

    def calculate_shifts(self, W, w):
        n = (W-1)//w + 1
        dt = 0
        if n > 1:
            dt = (w*n - W)//(n - 1)
            if dt < 10:
                n = n + 1
            dt = w - (w*n - W)//(n - 1)
        return dt, n

    def split_image_to_tiles(self, set_path):
        image_tiles = []
        for image_filename in os.listdir(set_path):
            img = cv2.imread(os.path.join(set_path, image_filename))
            W, H, _ = img.shape
            w, h, _ = self.tile_shape
            dx, nx = self.calculate_shifts(W, w)
            dy, ny = self.calculate_shifts(H, h)

            for x in range(nx):
                for y in range(ny):
                    ix = x*dx
                    iy = y*dy
                    sx = W - ix
                    sy = H - iy
                    if sx > w:
                        sx = w
                    if sy > h:
                        sy = h
                    image_tile = np.zeros(self.tile_shape, dtype=np.uint8)
                    image_tile[:sx, :sy] = img[ix:ix+sx, iy:iy+sy]
                    image_tiles.append(image_tile)

        return image_tiles

    def generator(self, mode, seed=1):
        if mode == 'test':
            image_gen = ImageDataGenerator()
            mask_gen = ImageDataGenerator()

            image_generator = image_gen.flow_from_directory(self.test_path, seed=seed, target_size=(256, 256),
                                                            class_mode='binary')
            mask_generator = mask_gen.flow_from_directory(self.test_mask_path, seed=seed, target_size=(256, 256),
                                                          class_mode='binary')

            data_generator = zip(image_generator, mask_generator)
            for (image, label) in data_generator:
                yield image, label

        if mode == 'val':
            image_gen = ImageDataGenerator()
            mask_gen = ImageDataGenerator()

            image_generator = image_gen.flow_from_directory(self.val_path, seed=seed, target_size=(256, 256),
                                                            class_mode='binary')
            mask_generator = mask_gen.flow_from_directory(self.val_mask_path, seed=seed, target_size=(256, 256),
                                                          class_mode='binary')

            data_generator = zip(image_generator, mask_generator)
            for (image, label) in data_generator:
                yield image, label

        if mode == 'train':
            image_gen = ImageDataGenerator(self.augmentation)
            mask_gen = ImageDataGenerator(self.augmentation)

            image_generator = image_gen.flow_from_directory(self.train_path, seed=seed, target_size=(256, 256),
                                                            class_mode='binary')
            mask_generator = mask_gen.flow_from_directory(self.train_mask_path, seed=seed, target_size=(256, 256),
                                                          class_mode='binary')

            data_generator = zip(image_generator, mask_generator)
            for (image, label) in data_generator:
                yield image, label