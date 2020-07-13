import cv2
import numpy as np


def main():

    input_video = 'input_video.avi'
    cap = cv2.VideoCapture(input_video)

    frame_count = 0
    ret, frame = cap.read()  # ret - bool==True, while frame is correct
    key = None

    # Variables for text

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    color_black = (0, 0, 0)

    # Variables for output video

    frame_shape = frame.shape
    frame_height = frame_shape[0]
    frame_width = frame_shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    output_video_without_noise = cv2.VideoWriter('output_video_without_noise.avi', fourcc, 20.0, (frame_width,
                                                                                                  frame_height), 0)

    output_video_with_box = cv2.VideoWriter('output_video_with_box.avi', fourcc, 20.0, (frame_width, frame_height), 0)

    while ret:

        # different color models

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mask for pink objects

        l_p = np.array([125, 60, 153])
        u_p = np.array([220, 210, 255])

        # mask for green objects

        l_g = np.array([(31, 32, 22)])
        u_g = np.array([65, 255, 255])

        # mask for yellow objects

        l_y = np.array([20, 45, 90])
        u_y = np.array([84, 255, 255])

        # mask for black objects

        l_b = np.array([5,5,5])
        u_b = np.array([180,255,45])

        # filters for masks

        mask_p_hsv = cv2.inRange(hsv, l_p, u_p)
        mask_g_hsv = cv2.inRange(hsv, l_g, u_g)
        mask_y_hsv = cv2.inRange(hsv, l_y, u_y)
        mask_b_hsv = cv2.inRange(hsv, l_b, u_b)

        mask_p_hsv_blur = cv2.medianBlur(mask_p_hsv, 3)
        mask_g_hsv_erode = cv2.erode(mask_g_hsv, kernel=np.ones((3,3), dtype=np.uint8))
        mask_g_hsv_blur = cv2.medianBlur(mask_g_hsv_erode, 3)
        mask_b_hsv_blur = cv2.medianBlur(mask_b_hsv, 3)
        mask_y_hsv_blur = cv2.medianBlur(mask_y_hsv, 3)

        # mask for box

        gray_frame_blur = cv2.GaussianBlur(gray_frame, (7, 7), 1)
        mask_box = cv2.adaptiveThreshold(gray_frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
        mask_box_grad = cv2.morphologyEx(mask_box, cv2.MORPH_GRADIENT, kernel=np.ones((7, 7), np.uint8))
        mask_box_blur = cv2.medianBlur(mask_box_grad, 7)

        all_masks = mask_p_hsv_blur + mask_g_hsv_blur + mask_b_hsv_blur + mask_y_hsv_blur + mask_box_blur
        masks_without_box = mask_p_hsv_blur + mask_g_hsv_blur + mask_b_hsv_blur + mask_y_hsv_blur

        final_mask = cv2.erode(all_masks, kernel=np.ones((7, 7), np.uint8))
        final_mask = cv2.medianBlur(final_mask, 5)  # a lot of noise

        final_mask_without_box = cv2.medianBlur(masks_without_box, 7)  # more clear

        # determine the shape of the object

        contours, _ = cv2.findContours(final_mask_without_box, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.arcLength(contour, closed=True) < 130:  # if perimeter > 130 => skip object
                continue

            perimeter = cv2.arcLength(contour, closed=True)  # perimeter of contour

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter,  True)  # second parameter in range 0.01 - 0.05

            # cv2.drawContours(final_mask, [approx], -1, color_black, 1)

            M = cv2.moments(contour)

            c_x = int(M["m10"] / M["m00"] // 1.05)  # slightly left of the center
            c_y = int(M["m01"] / M["m00"])

            if len(approx) == 3:
                cv2.putText(final_mask_without_box, 'Triangle', (c_x, c_y), font, font_size, color_black)

            elif len(approx) == 4:

                x1, y1, w, h = cv2.boundingRect(approx)  # get x and y, width and height
                aspect_ratio = float(w) / h

                if 0.9 <= aspect_ratio <= 1.1:
                    cv2.putText(final_mask_without_box, 'Square', (c_x, c_y), font, font_size, color_black)
                else:
                    cv2.putText(final_mask_without_box, 'Rectangle', (c_x, c_y), font, font_size, color_black)

            elif len(approx) > 7:

                cv2.putText(final_mask_without_box, 'Circle', (c_x, c_y), font, font_size, color_black)


        # Display 2 final results

        cv2.imshow('final', final_mask)
        cv2.imshow('final without box', final_mask_without_box)

        # Save frames

        output_video_with_box.write(final_mask)
        output_video_without_noise.write(final_mask_without_box)

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
    output_video_with_box.release()
    output_video_without_noise.release()


if __name__ == '__main__':
    main()


