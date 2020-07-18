import cv2
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.6

color_white = (255, 255, 255)
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_yellow = (0, 255, 255)


def check_distance(approx):

    ''' The idea is to calculate distance between contours points and then find
    difference between the longest and the shortest length between points to avoid
    strange shapes. '''

    length = len(approx)
    status = True
    all_dist = []

    if length >= 3:
        for i in range(length + 1):

            if i == len(approx):
                continue

            elif i == 0:
                dist = np.linalg.norm(approx[[0]] - approx[[1]])
                all_dist.append(dist)

            else:
                dist = np.linalg.norm(approx[[i - 1]] - approx[[i]])
                all_dist.append(dist)

        all_dist.sort()

        # print(all_dist)

        # difference between the longest and the shortest length between points

        difference = all_dist[-1] - all_dist[0]
        # print(difference)

        # 1.8 - threshold helps determine the circles shape, because they have a lot of points and distance
        # between them is small.

        if difference > all_dist[0] * 1.8:
            status = False

    return status


def shape_detector(frame, masks):

    color = None

    # get the color from the corresponding mask

    for key, value in masks.items():

        if key == 'red':
            color = color_red

        elif key == 'green':
            color = color_green

        elif key == 'yellow':
            color = color_yellow

        elif key == 'black':
            color = color_white

        contours, hier = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            perimeter = cv2.arcLength(contour, closed=True)  # perimeter of contour
            epsilon = 0.04 * perimeter

            approx = cv2.approxPolyDP(contour, epsilon, True)  # second parameter in range 0.01 - 0.05
            # approx - approximate a contour shape to another shape with less number of vertices depending

            if check_distance(approx) == False:
                continue

            x1, y1, w, h = cv2.boundingRect(approx)  # get x and y, width and height

            # to skip uncertain class

            if 0 <= x1 <= 25 or 60 <= y1 <= 75 or 615 <= x1 <= 640 or 380 <= y1 <= 420:
                continue

            M = cv2.moments(contour)  # gives a dictionary of all moment values calculated

            # x = int(M["m10"] / M["m00"] // 1.05)  # slightly left of the center
            # y = int(M["m01"] / M["m00"] // 1.2)  # over object

            x = approx.ravel()[0]  # coordinate on which print text of the shape
            y = approx.ravel()[1]

            if cv2.arcLength(contour, closed=True) < 110:  # if perimeter < 110 => skip object
                continue

            if key == 'yellow' and perimeter < 400:  # if perimeter > 350 => skip little yellow objects
                continue

            if len(approx) == 3:

                # this is a small crutch, it will work fine without it, if you want check it, just comment 2 lines
                if key == 'black' or key == 'yellow':  # to avoid black and yellow triangles :)
                    continue

                cv2.drawContours(frame,  [approx], -1, color, 2)
                cv2.putText(frame, 'Triangle', (x, y), font, font_size, color)

            elif len(approx) == 4:

                cv2.drawContours(frame, [approx], -1, color, 2)
                x1, y1, w, h = cv2.boundingRect(approx)  # get x and y, width and height
                aspect_ratio = w / h

                # print(aspect_ratio)
                if 0.95 <= aspect_ratio <= 1.05:
                    cv2.putText(frame, 'Square', (x, y), font, font_size, color)
                else:
                    cv2.putText(frame, 'Rectangle', (x, y), font, font_size, color)

            elif len(approx) < 10:

                # this is a small crutch, it will work fine without it, if you want check it, just comment 2 lines
                if key == 'yellow':  # to avoid yellow circles :)
                    continue

                epsilon = 0.01 * perimeter  # different epsilon for circles for smooth contours
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if check_distance(approx) == False:
                    continue

                cv2.drawContours(frame, [approx], -1, color, 2)
                cv2.putText(frame, 'Circle', (x, y), font, font_size, color)

            else:
                continue

    return frame


def main():

    input_video = 'input_video.avi'
    cap = cv2.VideoCapture(input_video)

    frame_count = 0
    ret, frame = cap.read()  # ret - bool==True, while frame is correct
    key = None

    # Variables for output video

    frame_shape = frame.shape
    frame_height = frame_shape[0]
    frame_width = frame_shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

    while ret:

        # different color models

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # mask for pink objects

        l_p = np.array([125, 40, 153])
        u_p = np.array([220, 210, 255])

        # mask for green objects

        l_g = np.array([(32, 40, 22)])
        u_g = np.array([75, 255, 255])

        # mask for yellow objects

        l_y = np.array([20, 45, 90])
        u_y = np.array([84, 255, 255])

        # mask for black objects

        l_b = np.array([5, 5, 5])
        u_b = np.array([180, 255, 38])

        mask_p = cv2.inRange(hsv, l_p, u_p)
        mask_g = cv2.inRange(hsv, l_g, u_g)
        mask_y = cv2.inRange(hsv, l_y, u_y)
        mask_b = cv2.inRange(hsv, l_b, u_b)

        # filters for masks

        mask_p = cv2.medianBlur(mask_p, 5)
        mask_g = cv2.medianBlur(mask_g, 5)
        mask_b_erode = cv2.erode(mask_b, kernel=np.ones((3, 3), dtype=np.uint8))
        mask_b = cv2.medianBlur(mask_b_erode, 5)

        mask_y = cv2.medianBlur(mask_y , 3)

        masks = [mask_p, mask_g, mask_b, mask_y]
        mask_color = ['red', 'green', 'black', 'yellow']

        masks_dict = dict(zip(mask_color, masks))

        # find candies

        frame = cv2.medianBlur(frame, 5)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rows = gray_frame.shape[0]
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=100, param2=30,
                                  minRadius=40, maxRadius=75)
        if circles is not None:

            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:

                center = (i[0], i[1])
                left_side = (int(i[0]*0.6), i[1])
                cv2.putText(frame, 'Who ate all candies?', left_side, font, font_size, color_red)

                # circle center

                cv2.circle(frame, center, 1, (0, 100, 100), 2)

                # circle outline

                radius = i[2]
                cv2.circle(frame, center, radius, (255, 0, 255), 2)

        shape_detector(frame, masks_dict)

        # Display final result

        cv2.imshow('frame', frame)

        # Save frame

        output_video.write(frame)

        # Pause on pressing of space.

        if key == ord(' '):
            wait_period = 0
        else:
            wait_period = 30

        # drawing, waiting, getting key, reading another frame

        key = cv2.waitKey(wait_period)
        ret, frame = cap.read()
        frame_count += 1

    cap.release()


if __name__ == '__main__':
    main()


