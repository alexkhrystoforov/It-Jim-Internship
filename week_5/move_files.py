import os
import shutil


def move_all_imgs():
    """
    move all images from dataset/*/ to all_img folder
    """

    folder_name = 'all_img'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        for root, dirs, files in os.walk('dataset'):
            for name in dirs + files:
                if root == 'dataset':
                    continue
                new_file_name = root[-3:] + '_' + name[-7:]
                shutil.move(root + '/' + name, 'all_img/' + new_file_name)

        # delete dataset folder
        shutil.rmtree('dataset')
    else:
        print('you have already create all_img')
