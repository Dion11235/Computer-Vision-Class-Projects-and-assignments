# Camera_Callibration

In this project, we use the geometric relationships between images taken from multiple views
to compute camera positions and estimate fundamental matrices for various scenes.Specifically
we will estimate the camera projection matrix or calibration matrix, which maps 3D world
coordinates to image coordinates, as well as the fundamental matrix, which relates points
in one scene to epipolar lines in another. The camera projection matrix and the fundamen-
tal matrix can each be estimated using point correspondences. To estimate the projection
matrix (camera calibration), the input is corresponding 3d and 2d points. To estimate the
fundamental matrix the input is corresponding 2d points across two images. We will start
out by estimating the projection matrix and the fundamental matrix for a scene with ground
truth correspondences. Then we eill move on to estimating the fundamental matrix using
point correspondences from ORB. By using RANSAC to find the fundamental matrix with
the most inliers, we can filter away spurious matches and achieve near perfect point to point
matching.

- you can find all the experiments and implementations in `proj3.ipynb` and `student_code.py` respectively inside `code` folder.
- you can find the outputs in the `results` folder corresponding to the images and data given in `data` folder.
