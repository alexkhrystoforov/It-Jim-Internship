The task is to track the object (marker.jpg) on the video (find_chocolate.mp4) using two approaches:
1) tracking by detection using orb features. You need to find homography and use it to draw a plane rectangle on the top of the marker so that it follows the orientation of the marker.
2) tracking using optical flow (Lucas-Kanade). In this case you initialize the tracker using the ORB features like in the above solution, but after that you update the positioning between the frames using optical flow. The output is to be drawn in the same way as in the approach 1. But it will have different behavior.

As an output you should provide two videos obtained using those two approaches.


Algorithm doesn't recognize object via ORB - 105 times but our plane rectangle on the object was clear.

bf_matches avg distance:  16.59372703784069
flann_matches avg distance 29.415291373025536
bf_knn_matches avg distance 19.867867188667198

This mean that bf_matches - are the best

Some thoughts about improving performance - template and frame preprocessing, deep tuning.

