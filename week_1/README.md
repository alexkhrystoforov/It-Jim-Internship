To make a function which enhances visibility of the objects in the video and creates a binary mask with them. The mask should be as clean as possible.
Input: video. Output: video.

You will probably need 
color space transform "cv2.cvtColor", 
threshold "cv2.threshold", 
morphology:
Histogram equalization
Adaptive threshold
Blur
Median
cv2.filter2D
