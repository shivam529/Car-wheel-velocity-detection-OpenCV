# Car-wheel-velocity-detection-OpenCV
Detecting wheels of cars from the input video and showing speed in pixels/frame.<br>

Given the assumptions:<br>
1) cars move from left to right<br>
2) cars don't overlap<br>
3) wheels are in the same plane throughout<br>
Method:<br>
I used OpenCV for hough lines.<br>
GrayScaled,Binary Thresholded, and Found Edges using canny edge operator.<br>
This edge image was used to find circles in.<br>

REMOVAL of false circles detected:<br>
Limted circles to empirically found vertical range of actual tires<br>
Using template matching(checking similarity) for a circle patch with a tire.<br>
Thid final step gave me perfect circles(wheels) for the given video.<br>

SPEED was calculated keeping a track of each wheel and X distance travelled along with no. of frames elapsed at each point.<br>

Output can be seen at the link: https://youtu.be/9YvrOLn3mqc <br>
Or be recreated by running the program again.


