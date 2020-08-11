import pandas as pd
import numpy as np
import cv2
import os
from skimage import feature
from skimage.filters import gabor_kernel
from sklearn.preprocessing import minmax_scale
from scipy import ndimage as nd


lbp_df = pd.DataFrame(columns=['lbp ' + str(i + 1) for i in range(258)])
hog_df = pd.DataFrame(columns=['hogs ' + str(i + 1) for i in range(256)])
gabor_df = pd.DataFrame(columns=['gabor ' + str(i + 1) for i in range(32)])

image_id = [i for i in range(1600)]
image_id = np.array(image_id)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + eps)

        return hist


def compute_feats(image, kernels):
    # feats = np.zeros((len(kernels), 2), dtype=np.double)
    descriptors = []
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        descriptors.append(filtered.mean())
        descriptors.append(filtered.var())
        # feats[k, 0] = filtered.mean()
        # feats[k, 1] = filtered.var()
    return descriptors


class Gradient_histogram:
    def __init__(self, numPoints):
        self.numPoints = numPoints

    def get_grad_features(self, grad_mag, grad_ang):
        angles = grad_ang[grad_mag > 5]
        hist, bins = np.histogram(angles, self.numPoints)

        return hist

    def describe(self, pattern_img, eps=1e-7):
        # Calculate Sobel gradient
        grad_x = cv2.Sobel(pattern_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(pattern_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag, grad_ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        hist = self.get_grad_features(grad_mag, grad_ang).astype(np.float32)
        return hist


images = load_images_from_folder('all_img')

kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # more features

    feature_extractor = LocalBinaryPatterns(256, 1)
    descriptor_template = feature_extractor.describe(img)
    descriptor_template_norm = minmax_scale(descriptor_template, feature_range=(0, 1))

    feature_extractor_2 = Gradient_histogram(256)
    descriptor_template_2 = feature_extractor_2.describe(img)
    descriptor_template_2_norm = minmax_scale(descriptor_template_2, feature_range=(0, 1))

    descriptor_template_3 = compute_feats(img, kernels)
    descriptor_template_3_norm = minmax_scale(descriptor_template_3, feature_range=(0, 1))

    gabor_df = gabor_df.append(pd.DataFrame(descriptor_template_3_norm.reshape(1, -1), columns=list(gabor_df)),
                           ignore_index=True)

    lbp_df = lbp_df.append(pd.DataFrame(descriptor_template_norm.reshape(1, -1), columns=list(lbp_df)),
                           ignore_index=True)
    hog_df = hog_df.append(pd.DataFrame(descriptor_template_2_norm.reshape(1, -1), columns=list(hog_df)),
                           ignore_index=True)

x_df = pd.concat([lbp_df, hog_df, gabor_df], axis=1, sort=False)

target = []

for dirname, _, filenames in os.walk('all_img'):
    for filename in filenames:
        target.append(filename[:3])

target = np.array(target)

y_df = pd.DataFrame(columns=['ImageId', 'target'])

y_df['ImageId'] = image_id
y_df['target'] = target

df = pd.concat([y_df, x_df], axis=1, sort=False)
# print(y_df.head())
# print(x_df.head())

df.to_csv('data.csv', encoding='utf-8', index=False)
