# dont watch only output_video_ORB

script running - show right understanding of lags(pauses) in video, its going on due to showing no object detecting on the frame.

Algorithm doesn't recognize object via ORB - 105 times but our plane rectangle on the object was clear.
I could did more deep tuning for happy mean

bf_matches avg distance:  16.59372703784069
flann_matches avg distance 29.415291373025536
bf_knn_matches avg distance 19.867867188667198

This mean that bf_matches - are the best

Some thoughts about improving performance - template and frame preprocessing, deep tuning.

Sorry for awful performance of OF, don't have enough time for this, but I'm going go deeper on OF at Wednesday