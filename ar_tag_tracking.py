"""
A tool for AR Tag position and orientation tracking using a camera feed

Dependencies:
 - cv2: https://pypi.org/project/opencv-python/ (pip install opencv-python)
 - cv2 contrib (pip install opencv-contrib-python)

Inspired by ar_track_alvar in ROS (http://wiki.ros.org/ar_track_alvar) and this Medium post showing several relevant
OpenCV commands (https://medium.com/lifeandtech/custom-ar-tag-detection-tracking-manipulation-d7543b8569ea)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def order(pts: np.array):
    """
    Returns the order of the points in the camera frame

    Taken from https://medium.com/lifeandtech/custom-ar-tag-detection-tracking-manipulation-d7543b8569ea
    :param pts: Numpy array of points (List[Tuple])
    :return: Numpy array of points, ordered in camera frame ((minX, minY), (minX, maxY), (maxX, maxY), (maxX, minY))
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    # print(np.argmax(s))
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # print(np.argmax(diff))
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# Function to compute homography between world and camera frame
def homography(p, p1):
    A = []
    p2 = order(p)

    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :]/Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    # print(l)
    # print(h)
    return h


def find_ar_tag(img, ref, hessian_threshold=400, ratio_threshold=0.75):
    """
    Uses SIFT + FLANN to detect an instance of the reference object in the provided image

    Reference: https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
    Can use either SIFT, SURF, or ORB: https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/
    :param img: The provided image, containing max one instance of the reference object
    :param ref: The reference image of the object
    :param hessian_threshold: Only features, whose hessian is larger than hessianThreshold are retained by the detector.
                              Therefore, the larger the value, the less keypoints you will get. Usu. 300-500.
    :param ratio_threshold: After finding matches using KNN, filter matches using the Lowe's ratio test (closer to 1 =
                            less restrictive)
    :return: (Bounding box of the reference image in the image (size 4x2),
              pixel location of the matched keypoints in source img (size # matches x 2),
              homography matrix (3x3))
    """
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    keypoints_img, descriptors_img = detector.detectAndCompute(img, None)
    keypoints_ref, descriptors_ref = detector.detectAndCompute(ref, None)

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_ref, descriptors_img, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh*n.distance:
            good_matches.append(m)

    # -- Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj[i, 0] = keypoints_ref[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = keypoints_ref[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = keypoints_img[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = keypoints_img[good_matches[i].trainIdx].pt[1]
    H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)

    # -- Get the corners from the ref ( the object to be "detected" )
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = ref.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = ref.shape[1]
    obj_corners[2, 0, 1] = ref.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = ref.shape[0]
    scene_corners = cv2.perspectiveTransform(obj_corners, H)

    return np.reshape(scene_corners, (4, 2)), scene, H
    # return [(obj_corners[0, 0, 0], obj_corners[0, 0, 1]), (obj_corners[1, 0, 0], obj_corners[1, 0, 1]),
    #         (obj_corners[2, 0, 0], obj_corners[2, 0, 1]), (obj_corners[3, 0, 0], obj_corners[3, 0, 1])], scene


def homography_to_axes(H: np.ndarray, K: np.array):
    """
    Convert a homography matrix into its local x, y, and z orientation axes relative to the camera

    Helpful sources:
    https://stackoverflow.com/questions/8927771/computing-camera-pose-with-homography-matrix-based-on-4-coplanar-points/10781165#10781165
    https://stackoverflow.com/questions/41526335/decompose-homography-matrix-in-opencv-python
    :param H: Homography matrix
    :param K: Intrinsic camera matrix
    :return: Axes of (one possible) set of x, y, and z axes that correspond to the given homography
    """
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

def compute_intrinsic_camera_matrix(img_size_x, img_size_y, fov_x, fov_y, axis_skew=0):
    """
    Computes the intrinsic camera matrix K

    Sources:
    https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html
    https://ksimek.github.io/2013/08/13/intrinsic/

    :param img_size_x: Number of pixels of the camera image in the x direction
    :param img_size_y: Number of pixels of the camera image in the y direction
    :param fov_x: Degrees of camera's rated field of view in the x direction
    :param fov_y: Degrees of camera's rated field of view in the y direction
    :param axis_skew: Usually zero, but sometimes affected by digital processing techniques
    :return: Intrinsic camera matrix K
    """
    img_size_x /= 2
    img_size_y /= 2
    f_x = img_size_x / np.tan(np.deg2rad(fov_x) / 2)  # focal length x
    f_y = img_size_y / np.tan(np.deg2rad(fov_y) / 2)  # focal length y
    return np.array([[f_x, axis_skew, img_size_x],
                     [0, f_y, img_size_y],
                     [0, 0, 1]])


if __name__ == '__main__':
    K = compute_intrinsic_camera_matrix(1920, 1080, fov_x=46.4*2, fov_y=29.1*2)  # camera matrix for E-Meet (https://smile.amazon.com/gp/product/B08DXSG5QR/)
    print(K)
