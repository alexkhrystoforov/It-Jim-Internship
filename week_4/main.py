from tkinter import Tk
from tkinter.filedialog import askopenfilename

# change the path!
from week_4.utils import *
from week_4.rename_files import rename_files
from week_4.lbp import *


# change folders and images names
sorted_by_class, all_img_names, folder_names = rename_files()

# create our Plotter obj
plotter = Plotter(sorted_by_class)

# PLOT CLASSES HISTOGRAMS
classes_bgr_hist = plotter.plot_3d_hist(color_space='bgr')
plotter.plot_3d_hist(color_space='bgr')
classes_hsv_hist = plotter.plot_3d_hist(color_space='hsv')
plotter.plot_3d_hist(color_space='hsv')

classes_r_hist = plotter.plot_1d_hist(channel='r')
plotter.plot_1d_hist(channel='r')
classes_g_hist = plotter.plot_1d_hist(channel='g')
plotter.plot_1d_hist(channel='g')
classes_b_hist = plotter.plot_1d_hist(channel='b')
plotter.plot_1d_hist(channel='b')
classes_hsv_v_hist = plotter.plot_1d_hist(channel='v')
plotter.plot_1d_hist(channel='v')

list_all_hists = [
    classes_bgr_hist,
    classes_hsv_hist,
    classes_r_hist,
    classes_g_hist,
    classes_b_hist,
    classes_hsv_v_hist
]

# read json with some useful data
with open('images_hists_sorted_by_class.json', 'r') as json_file:
    images_hists_sorted_by_class = json.load(json_file)
    images_hists_sorted_by_class = json.loads(images_hists_sorted_by_class)

with open('all_images_hists.json', 'r') as json_file:
    all_images_hists = json.load(json_file)
    all_images_hists = json.loads(all_images_hists)


# MEAN AND MEDIAN OF CORRELATIONS OF CLASS HISTOGRAM AND ALL IMAGES HISTOGRAMS IN CLASS
# cv2.HISTCMP_CORREL - works here the best, comparing correlation with other opencv and scipy methods
statistic_dict = filter_for_hist_method(images_hists_sorted_by_class, folder_names, list_all_hists)

print('MEAN AND MEDIAN OF CORRELATIONS OF CLASS HISTOGRAM AND ALL IMAGES HISTOGRAMS IN CLASS')
# print('mean and median for brg hists ', statistic_dict.get('0'))
# print('mean and median for hsv hists ', statistic_dict.get('1'))
# print('mean and median for r channel hists ', statistic_dict.get('2'))
# print('mean and median for g channel hists ', statistic_dict.get('3'))
print('mean and median for b channel hists ', statistic_dict.get('4'))
# print('mean and median for v channel hists ', statistic_dict.get('5'))

# Lets explore our results and make the conclusion.
# We have the biggest correlation in brg colorspace and b channel, they are almost the same.
# Lets chose the b channel for further comparing and for distributing our random image in concrete class


Tk().withdraw()
filename = askopenfilename()


our_image_for_search = cv2.imread(filename)
hist = cv2.calcHist(our_image_for_search, [0], None, [256], [0, 256])

statistic_of_random_img = {}

status_hist_method, potentional_folders = method_hist(our_image_for_search, folder_names, statistic_dict, classes_b_hist)

if status_hist_method:
    # Works good only for images, where color is obviously good feature, for example:
    # 780/028.jpg , 792/209.jpg and etc
    print('Our method is histogram comparing')
    print('potentional folders for our img is ', potentional_folders)
    images_indexes = get_top_5_images_from_class(potentional_folders, hist)

    show_common_images(images_indexes, potentional_folders, folder_names, sorted_by_class)
else:
    print('in developing...')

