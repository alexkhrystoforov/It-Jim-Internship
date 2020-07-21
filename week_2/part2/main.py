import cv2
import glob
import numpy as np
import json

# variables for text, colors

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.6
color_black = (0, 0, 0)


class Template:

    def __init__(self, template):
        self.template = template
        self.rotate_90_counterclockwise = cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.rotate_90_clockwise = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
        self.rotate_180 = cv2.rotate(template, cv2.ROTATE_180)


def main():

    image_filename = 'plan.png'
    img = cv2.imread(image_filename)

    # I tried to do matching with gray frame, but it doesnt make score better

    # gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, gray_frame_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]

    # cover for extra symbols, if need it

    # white_cover = cv2.rectangle(img, (3610, 1650), (4300, 3000), (255, 255, 255), -1)
    # white_cover_gray_frame = cv2.rectangle(gray_frame, (3610, 1650), (4300, 3000), (255, 255, 255), -1)

    file_names = glob.glob("symbols/*.png")  # store images names  to list
    file_names.sort()  # sort images names in right order
    symbols = [cv2.imread(img) for img in file_names]

    # 1 template

    template_1 = Template(symbols[0])
    template_1_thresh = 0.9
    template_1_symbol_description = 'Power Receptacle'

    # template augmentation

    template_1_augmentation_1 = template_1.rotate_90_clockwise.copy()
    template_1_augmentation_1 = cv2.rectangle(template_1_augmentation_1, (9, 0), (13, 19), color_black, -1)

    all_templates_1 = [template_1.template, template_1.rotate_90_clockwise, template_1.rotate_90_counterclockwise,
                      template_1.rotate_180, template_1_augmentation_1]

    # 2 template

    template_2 = Template(symbols[1])
    template_2_thresh = 0.99
    template_2_symbol_description = 'Power Receptacle'

    # template augmentation

    template_2_copy = template_2.template.copy()
    template_2_augmentation_1 = cv2.line(template_2_copy, (2, 10), (6, 10), color_black)

    template_2_copy = template_2.template.copy()
    template_2_augmentation_2 = cv2.line(template_2_copy, (0, 5), (18, 5), color_black, 2)
    template_2_augmentation_2 = cv2.line(template_2_augmentation_2, (1, 19), (1, 24), color_black)

    template_2_rotate_90_counterclockwise_copy = template_2.rotate_90_counterclockwise.copy()
    template_2_augmentation_3 = cv2.line(template_2_rotate_90_counterclockwise_copy, (17, 17), (17, 4), color_black)
    template_2_augmentation_3 = cv2.rectangle(template_2_augmentation_3, (2, 0), (2, 17), color_black, 2)

    template_2_rotate_180_copy = template_2.rotate_180.copy()
    template_2_augmentation_4 = cv2.line(template_2_rotate_180_copy, (5, 23), (18, 23), color_black)
    template_2_augmentation_4 = cv2.line(template_2_augmentation_4, (0, 38), (18, 38), color_black, 2)

    all_templates_2 = [template_2.template, template_2.rotate_90_counterclockwise, template_2.rotate_180,
                      template_2_augmentation_1, template_2_augmentation_2, template_2_augmentation_3,
                      template_2_augmentation_4]

    # 3 template

    template_3 = Template(symbols[2])
    template_3_thresh = 0.88
    template_3_symbol_description = 'Power Receptacle'

    all_templates_3 = [template_3.template]

    # 4 template

    template_4 = Template(symbols[3])
    template_4_thresh = 0.999
    template_4_symbol_description = 'Power Receptacle'

    all_templates_4 = [template_4.template, template_4.rotate_90_counterclockwise]

    # 5 template

    template_5 = Template(symbols[4])
    template_5_thresh = 0.99
    template_5_symbol_description = 'Power Receptacle'

    all_templates_5 = [template_5.template]

    # 6 template

    template_6 = Template(symbols[5])
    template_6_thresh = 0.9
    template_6_symbol_description = 'Floor Core'

    all_templates_6 = [template_6.template]

    # 7 template

    template_7 = Template(symbols[6])
    template_7_thresh = 0.9
    template_7_symbol_description = 'Floor Work'

    all_templates_7 = [template_7.template]

    # 8 template

    template_8 = Template(symbols[7])
    template_8_thresh = 0.93
    template_8_symbol_description = 'Communication'

    # template augmentation

    template_8_rotate_90_counterclockwise_copy = template_8.rotate_90_counterclockwise.copy()
    template_8_augmentation_1 = cv2.line(template_8_rotate_90_counterclockwise_copy, (11, 23), (11, 4), color_black)
    template_8_rotate_90_counterclockwise_copy = template_8.rotate_90_counterclockwise.copy()
    template_8_augmentation_2 = cv2.line(template_8_rotate_90_counterclockwise_copy, (11, 17), (11, 0), color_black)

    template_8_rotate_90_clockwise_copy = template_8.rotate_90_clockwise.copy()
    template_8_augmentation_3 = cv2.line(template_8_rotate_90_clockwise_copy, (14, 0), (14, 23), color_black)
    template_8_rotate_90_clockwise_copy = template_8.rotate_90_clockwise.copy()
    template_8_augmentation_4 = cv2.rectangle(template_8_rotate_90_clockwise_copy, (0, 0), (10, 8), color_black, -1)

    all_templates_8 = [template_8.template, template_8.rotate_90_counterclockwise, template_8.rotate_90_clockwise,
                      template_8_augmentation_1, template_8_augmentation_2, template_8_augmentation_3,
                      template_8_augmentation_4 ]

    # 9 template

    template_9 = Template(symbols[8])
    template_9_thresh = 0.95
    template_9_symbol_description = 'Communication'

    all_templates_9 = [template_9.template, template_9.rotate_90_counterclockwise]

    # 10 template

    template_10 = Template(symbols[9])
    template_10_thresh = 0.94
    template_10_symbol_description = 'Junction Box'

    # template augmentation

    template_10_1_object_for_crop = np.zeros((26, 12, 3), np.uint8)
    template_10_1_object_for_crop[:] = (255, 255, 255)

    template_10_1_object_for_crop = cv2.line(template_10_1_object_for_crop, (0, 13), (12, 13), color_black)
    template_10_crop = template_10.template[0:26, 0:26]

    template_10_augmentation_1 = np.concatenate((template_10_1_object_for_crop, template_10_crop), axis=1)
    template_10_augmentation_2 = np.concatenate((template_10_crop, template_10_1_object_for_crop), axis=1)

    all_templates_10 = [template_10.template, template_10_augmentation_1, template_10_augmentation_2]

    # 11 template

    template_11 = Template(symbols[10])
    template_11_thresh = 0.9
    template_11_symbol_description = 'Recessed TV'

    all_templates_11 = [template_11.template]

    # 12 template

    template_12 = Template(symbols[11])
    template_12_thresh = 0.97
    template_12_symbol_description = 'Card Reader'

    # template augmentation

    template_12_1_object_for_crop = np.zeros((13, 26, 3), np.uint8)
    template_12_1_object_for_crop[:] = (255, 255, 255)

    template_12_2_object_for_crop = np.zeros((2, 26, 3), np.uint8)
    template_12_2_object_for_crop[:] = (255, 255, 255)

    template_12_3_object_for_crop = np.zeros((26, 13, 3), np.uint8)
    template_12_3_object_for_crop[:] = (255, 255, 255)

    template_12_1_object_for_crop = cv2.line(template_12_1_object_for_crop, (13, 0), (13, 13), color_black)
    template_12_2_object_for_crop = cv2.line(template_12_2_object_for_crop, (1, 1), (26, 1), color_black, 2)
    template_12_3_object_for_crop = cv2.line(template_12_3_object_for_crop, (0, 13), (13, 13), color_black, 2)

    template_12_crop = template_12.template[0:26, 0:26]

    template_12_augmentation_1 = np.concatenate((template_12_1_object_for_crop, template_12_crop))
    template_12_augmentation_1 = np.concatenate((template_12_2_object_for_crop, template_12_augmentation_1))

    template_12_augmentation_2 = np.concatenate((template_12_crop, template_12_1_object_for_crop))

    template_12_augmentation_3 = np.concatenate((template_12_3_object_for_crop, template_12_crop), axis=1)

    all_templates_12 = [template_12.template, template_12_augmentation_1, template_12_augmentation_2,
                       template_12_augmentation_3]

    # 13 template

    template_13 = Template(symbols[12])
    template_13_thresh = 0.875
    template_13_symbol_description = 'Push To Exit'

    all_templates_13 = [template_13.template]

    # 14 template

    template_14 = Template(symbols[13])  # symbol is a little bit broken, we should change it
    template_14_thresh = 0.99
    template_14_symbol_description = 'Blank'

    # template augmentation

    template_14_crop = template_14.template[0:40, 0:26]

    template_14_1_object_for_crop = np.zeros((40, 1, 3), np.uint8)  # vertical line on the right of template
    template_14_1_object_for_crop[:] = (255, 255, 255)

    template_14_2_object_for_crop = np.zeros((1, 27, 3), np.uint8)  # horizontal line under template
    template_14_2_object_for_crop[:] = (255, 255, 255)

    # vertical line on the right of template
    template_14_1_object_for_crop = cv2.line(template_14_1_object_for_crop, (0, 0), (0, 25), color_black)

    # horizontal line under template
    template_14_2_object_for_crop = cv2.line(template_14_2_object_for_crop, (0, 0), (27, 0), color_black)

    template_14_augmentation_1 = np.concatenate((template_14_crop, template_14_1_object_for_crop), axis=1)
    template_14_augmentation_1 = np.concatenate((template_14_augmentation_1, template_14_2_object_for_crop))

    all_templates_14 = [template_14_augmentation_1]  # drop original template

    # 15 template

    template_15 = Template(symbols[14])
    template_15_thresh = 0.95
    template_15_symbol_description = 'Fire Safety'

    # template augmentation

    template_15_crop = template_15.template[0:49, 0:23]

    template_15_1_object_for_crop = np.zeros((2, 23, 3), np.uint8)  # horizontal line under template
    template_15_1_object_for_crop[:] = (255, 255, 255)

    # horizontal line under template
    template_15_1_object_for_crop = cv2.rectangle(template_15_1_object_for_crop, (0, 0), (23, 2), color_black, -1)

    template_15_augmentation_1 = np.concatenate((template_15_crop, template_15_1_object_for_crop))

    all_templates_15 = [template_15.template, template_15.rotate_90_counterclockwise, template_15.rotate_90_clockwise,
                        template_15.rotate_180, template_15_augmentation_1]

    # 16 template

    template_16 = Template(symbols[15])
    template_16_thresh = 0.95
    template_16_symbol_description = 'Key - Note'

    all_templates_16 = [template_16.template]

    # Store our data in lists

    all_templates = []
    all_templates.extend(value for name, value in locals().items() if name.startswith('all_templates_'))

    symbols_description = []
    symbols_description.extend(value for name, value in locals().items() if name.endswith('_symbol_description'))

    templates_thresh = []
    templates_thresh.extend(value for name, value in locals().items()
                            if name.endswith('_thresh') and name.startswith('template_'))

    # When I tried match all templates separately (use code below), I had a much better result than the final.
    # Dont know why.
    # Maybe due to color rectangle and symbols description.
    # I tried do match separately doing it on img.copy() instead of img in detect_template() function and then
    # doing bitwise_and
    # But had the same result

    # random_color = list(np.random.random(size=3) * 256)
    #
    # for template in all_templates_15:
    #
    #     matching = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    #     matching = cv2.normalize(matching, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    #     thresh = cv2.threshold(matching, np.max(matching) * 0.95, 255, cv2.THRESH_BINARY)[1]
    #     conts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     for cnt in conts:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         cv2.rectangle(img, (x + w, y + h), (x + w + template.shape[1], y + h +
    #                                             template.shape[0]), random_color, 1)
    #         cv2.putText(img, symbols_description[4], (x, y), font, font_size, random_color)
    #         # cv2.drawContours(img, conts, -1, [0, 0, 255], 15)
    #
    #     cv2.namedWindow('Window with example', cv2.WINDOW_NORMAL)
    #     cv2.imshow('Window with example', img)
    #     cv2.waitKey(0)

    detect_template(all_templates, symbols_description, templates_thresh)


def detect_template(templates, symbols_description, templates_thresh):

    image_filename = 'plan.png'
    img = cv2.imread(image_filename)
    data_for_json = {}

    for i in range(len(templates)):

        random_color = list(np.random.random(size=3) * 256)

        for j in range(len(templates[i])):
            matching = cv2.matchTemplate(img, templates[i][j], cv2.TM_CCOEFF_NORMED)
            matching = cv2.normalize(matching, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            thresh = cv2.threshold(matching, np.max(matching) * templates_thresh[i], 255, cv2.THRESH_BINARY)[1]
            conts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            data_for_json[symbols_description[i]] = []

            for cnt in conts:
                x, y, w, h = cv2.boundingRect(cnt)

                # append matched template coordinates to dict

                data_for_json[symbols_description[i]].append({
                    'top_left': [x, y],
                    'top_right': [x, y + templates[i][j].shape[0]],
                    'bottom_left': [x + templates[i][j].shape[1], y],
                    'bottom_right': [x + templates[i][j].shape[1], y + templates[i][j].shape[0]]
                })

                cv2.rectangle(img, (x + w, y + h),
                                            (x + w + templates[i][j].shape[1], y + h + templates[i][j].shape[0]),
                                            random_color, 1)
                cv2.putText(img, symbols_description[i], (x, y), font, font_size, random_color)

                # The idea is to avoid double matching, check is any pixel of concrete random_color
                # in nearby area. But it doesnt work... :(

                # patch = img[y-5:y+5, x-5:x+5]

                # if random_color not in patch:

                # cv2.rectangle(img, (x + w, y + h),
                #               (x + w + templates[i][j].shape[1], y + h + templates[i][j].shape[0]),
                #               random_color, 1)
                # cv2.putText(img, symbols_description[i], (x, y), font, font_size, random_color)

    with open('data.txt', 'w') as outfile:
        json.dump(data_for_json, outfile)

    cv2.namedWindow('Window with example', cv2.WINDOW_NORMAL)
    cv2.imshow('Window with example', img)
    cv2.waitKey(0)

    cv2.imwrite('new_plan.png', img)


if __name__ == '__main__':
    main()
