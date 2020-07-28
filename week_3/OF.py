import cv2
import numpy as np
from week_3.utils import *
from week_3.ORB import *


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=500,
                          qualityLevel=0.2,
                          minDistance=50,
                          blockSize=10)

lk_params = dict(winSize=(15, 15),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0,255,0)


def main():

    input_video = 'find_chocolate.mp4'
    cap = cv2.VideoCapture(input_video)

    ret, frame = cap.read()

    marker_name = 'marker.jpg'
    marker = cv2.imread(marker_name)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter('output_video_OF.avi', fourcc, 30.0, (frame.shape[1], frame.shape[0]))

    frame_count = 0
    key = None

    # Take first frame
    first_ret, first_frame = cap.read()
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # ORB detector

    orb = cv2.ORB_create(nfeatures=1000, WTA_K=3)  # better than WTA_K=4

    # Initialize BF matcher

    matcher = Matcher(orb, first_frame_gray, marker)
    bf_matches = matcher.BF_matcher()

    src_pts = np.float32([matcher.kp_marker[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([matcher.kp_frame[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Perspective transform

    h, w = marker.shape[:2]
    first_pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

    first_dst = cv2.perspectiveTransform(first_pts, matrix)

    homography = cv2.polylines(frame, [np.int32(first_dst)], True, color, 3, cv2.LINE_AA)

    mask = np.zeros_like(first_frame)
    first_pts = cv2.goodFeaturesToTrack(first_frame_gray, mask=None, **feature_params)

    while ret:
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        new_pts, status, error = cv2.calcOpticalFlowPyrLK(homography, frame, first_pts, None, **lk_params)

        good_new = new_pts[status == 1]
        good_old = first_pts[status == 1]

        # draw the tracks
        # count_of_moved = 0
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            velocity = np.sqrt((a - c) ** 2 + (b - d) ** 2)
            if velocity > 1:
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                frame = cv2.circle(frame, (int(a), int(b)), 4, color, -1)

        query_pts = good_old.reshape(-1, 1, 2)
        train_pts = good_new.reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        dst = cv2.perspectiveTransform(first_dst, matrix)

        homography = cv2.polylines(frame, [np.int32(dst)], True, color, 3)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', homography)

        # Now update the previous frame and previous points

        first_dst = dst
        first_pts = good_new.reshape(-1, 1, 2)
        first_frame_gray = frame_gray

        # Save frame

        output_video.write(homography)

        # Pause on pressing of space.
        if key == ord(' '):
            wait_period = 0
        else:
            wait_period = 30

        # drawing, waiting, getting key, reading another frame

        key = cv2.waitKey(wait_period)
        ret, frame = cap.read()

    cap.release()


if __name__ == '__main__':
    main()
