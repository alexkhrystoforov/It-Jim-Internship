import cv2
import glob
import numpy as np


def show_video(folder_name):
    """
    Show frame + mask in one video ( stacked horizontally )
    """
    all_filenames = []
    all_masks_filenames = []

    for filename in glob.glob('foosball_dataset/' + folder_name + '/*'):
        all_filenames.append(filename)

    for filename in glob.glob('foosball_dataset/' + folder_name + '_mask/*'):
        all_masks_filenames.append(filename)

    all_filenames = sorted(all_filenames)
    all_masks_filenames = sorted(all_masks_filenames)

    key = None

    for i in range(len(all_filenames)):
        img1 = cv2.imread(all_filenames[i])
        img2 = cv2.imread(all_masks_filenames[i])

        stack = np.concatenate((img1, img2), axis=1)
        cv2.imshow('video', stack)

        if key == ord(' '):
            wait_period = 0
        else:
            wait_period = 4

        key = cv2.waitKey(wait_period)


show_video('train_set')
# show_video('val_set')
# show_video('test_set')
