import cv2
import numpy as np
import random
from week_3.utils import *


font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 2
color_red = (0, 0, 255)

class Matcher:

    def __init__(self, orb,  frame, marker):

        self.kp_frame, self.des_frame = orb.detectAndCompute(frame, None)
        self.kp_marker, self.des_marker = orb.detectAndCompute(marker, None)
        self.norm_hamming = cv2.NORM_HAMMING2

    def BF_matcher(self):
        bf = cv2.BFMatcher(self.norm_hamming, crossCheck=True)
        matches = bf.match(self.des_marker, self.des_frame)
        matches = sorted(matches, key=lambda x: x.distance)  # sort matches
        best_matches = find_best_matches(matches)
        return best_matches

    def FLANN_matcher(self):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.des_frame, self.des_marker, k=2)
        best_matches = lowe_ratio(matches, ratio_thresh=0.8)
        return best_matches

    def BF_matcher_knn(self, k=2):
        bf = cv2.BFMatcher(self.norm_hamming, crossCheck=False)  # crosscheck - define if you have 1 match
        matches = bf.knnMatch(self.des_frame, self.des_marker, k)
        matches = sorted(matches, key=lambda x: x[0].distance)
        # Lowe's ratio test
        best_matches = [[m] for (m, n) in matches if m.distance < 0.75 * n.distance]

        return best_matches


def main():

    input_video = 'find_chocolate.mp4'
    cap = cv2.VideoCapture(input_video)

    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 901 frames

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter('output_video_ORB.avi', fourcc, 20.0, (3680, 720))

    # bbox = cv2.selectROI(frame, False)

    marker_name = 'marker.jpg'
    marker = cv2.imread(marker_name)

    MIN_MATCH_COUNT = 45
    frame_count = 0
    no_object_count = 0
    key = None

    # ORB detector

    orb = cv2.ORB_create(nfeatures=1000,  WTA_K=3)  # better than WTA_K=4

    # create random frame set 20% from original frames number

    random_frames = set()
    sample_size = total_frames // 5
    answer_size = 0

    while answer_size < sample_size:
        r = random.randint(0, 901)
        if r not in random_frames:
            answer_size += 1
            random_frames.add(r)

    matcher1_top100 = []
    matcher2_top100 = []
    matcher3_top100 = []

    matcher1_mean_list = []
    matcher2_mean_list = []
    matcher3_mean_list = []

    while ret:

        matcher = Matcher(orb, frame, marker)

        bf_matches = matcher.BF_matcher()
        bf_knn_matches = matcher.BF_matcher_knn()
        flann_matches = matcher.FLANN_matcher()

        # check matchers quality. totally we have 901 frames, lets get randomly 20% frames and compare matchers quality

        # sorry for this hardcoding below, dont have enough time for refactoring

        current_frame_matches = get_100_best_matches(bf_matches, flann_matches, bf_knn_matches)

        if frame_count in random_frames:
            matcher1_top100.append(current_frame_matches[0])
            matcher2_top100.append(current_frame_matches[1])
            matcher3_top100.append(current_frame_matches[2])

        matcher1_mean, matcher2_mean, matcher3_mean = get_mean(matcher1_top100, matcher2_top100, matcher3_top100)

        if frame_count in random_frames:
            matcher1_mean_list.append(matcher1_mean)
            matcher2_mean_list.append(matcher2_mean)
            matcher3_mean_list.append(matcher3_mean)

        if len(matcher1_mean_list) >= 2:
            matcher1_mean_list.pop(0)
            matcher2_mean_list.pop(0)
            matcher3_mean_list.pop(0)

        # print avg distance for every matcher

        if frame_count == total_frames-1:
            print('bf_matches avg distance: ', statistics.mean(matcher1_mean_list[0]))
            print('flann_matches avg distance', statistics.mean(matcher2_mean_list[0]))
            print('bf_knn_matches avg distance', statistics.mean(matcher3_mean_list[0]))

        # We conclude that bf_matches find the best matches

        # extract the matched keypoints

        if len(bf_matches) > MIN_MATCH_COUNT:

            src_pts = np.float32([matcher.kp_marker[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([matcher.kp_frame[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Perspective transform

            h, w = marker.shape[:2]
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, matrix)

            if check_points(dst):

                draw_params = dict(outImg=None,
                                   matchColor=(0, 255, 0),
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                result_mathing_bf = cv2.drawMatches(marker, matcher.kp_marker, frame, matcher.kp_frame, bf_matches,
                                                    **draw_params)

                homography = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # testing warps for myself
                max_extent = np.max(pts, axis=0)[0].astype(np.int)[::-1]
                sz_out = (max(max_extent[1], marker.shape[1]), max(max_extent[0], marker.shape[0]))
                warped = cv2.warpPerspective(marker, matrix, dsize=sz_out)

                separetor = np.zeros((720, 560, 3), np.uint8)
                separetor[:] = (255, 255, 255)
                homography = np.concatenate((homography, separetor), axis=1)
                result_mathing_bf = cv2.resize(result_mathing_bf, (1840, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

                stack = np.concatenate((homography, result_mathing_bf), axis=1)

                cv2.namedWindow('homography + matches', cv2.WINDOW_NORMAL)
                cv2.imshow('homography + matches', stack)
            else:
                no_object_count += 1
                cv2.putText(frame, 'dont recognize ' + str(no_object_count) + ' times', (100, 100), font, font_size, color_red,
                            thickness=5)
                cv2.namedWindow('no object', cv2.WINDOW_NORMAL)
                cv2.imshow("no object", frame)

        else:

            # show the frame where we don't track the object ( and we understand why we dont track )
            no_object_count += 1
            cv2.putText(frame, 'dont recognize ' + str(no_object_count) + ' times', (100, 100), font, font_size, color_red,
                        thickness=5)
            cv2.namedWindow('no object', cv2.WINDOW_NORMAL)
            cv2.imshow("no object", frame)

        # Save frame

        output_video.write(stack)

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


