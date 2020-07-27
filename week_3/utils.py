import numpy as np
import statistics


def find_best_matches(matches):
    """
    Filter matches by distance
    Args:
         matches: list
    Returns:
        best_matches: list
    """
    best_matches = []
    for m in matches:
        if m.distance < 25:
            best_matches.append(m)

    return best_matches


def lowe_ratio(matches, ratio_thresh):
    """
    Filter matches using the Lowe's ratio test
    Args:
        matches: list
        ratio_thresh: float
    Returns:
        best_matches: list
     """

    best_matches = []

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                best_matches.append(m)

        except ValueError:
            pass

    return best_matches


def get_100_best_matches(matcher1, matcher2, matcher3):
    """
        Get best 100 matches from each matcher

        Args:
            matcher1: list
            matcher2: list
            matcher3: list
        Returns:
            matcher1_first_100: list
            matcher2_first_100: list
            matcher3_first_100: list
        """
    matcher1_first_100 = []
    matcher2_first_100 = []
    matcher3_first_100 = []

    for i in range(len(matcher1[:100])):
        matcher1_first_100.append(matcher1[i].distance)

    for i in range(len(matcher2[:100])):
        matcher2_first_100.append(matcher2[i].distance)

    for i in range(len(matcher3[:100])):
        matcher3_first_100.append(matcher3[i][0].distance)

    return matcher1_first_100, matcher2_first_100, matcher3_first_100


def get_mean(matcher1, matcher2, matcher3):

    matcher1_mean = []
    matcher2_mean = []
    matcher3_mean = []

    for i in range(len(matcher1)):
        matcher1_mean.append(statistics.mean(matcher1[i]))
        matcher2_mean.append(statistics.mean(matcher2[i]))
        matcher3_mean.append(statistics.mean(matcher3[i]))

    return matcher1_mean, matcher2_mean, matcher3_mean


def check_points(dst):
    ''' The idea is to calculate distance between object corners points and then find
        difference between the longest and the shortest length between points to avoid
        strange shapes. '''
    status = True
    all_dist = []

    if len(dst) == 4:
        for i in range(len(dst)):

            if i == 0:
                dist = np.linalg.norm(dst[[0]] - dst[[1]])
                all_dist.append(dist)
            else:
                dist = np.linalg.norm(dst[[i - 1]] - dst[[i]])
                all_dist.append(dist)

        all_dist.sort()

        # difference between the longest and the shortest length between points
        difference = all_dist[-1] - all_dist[0]

        if difference > all_dist[0] * 1.9 or min(all_dist) < 20:
            status = False

    return status

