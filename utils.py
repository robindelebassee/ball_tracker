import cv2
import math
import numpy as np


def stabilize(frame, prev_gray, curr_gray, warp_matrix, warp_mode, termination_criteria):

    # Compute the homographic matrix between the previous and the current grayscale frames using EEC algorithm
    _, warp_matrix = cv2.findTransformECC(
        prev_gray, curr_gray, warp_matrix, warp_mode, termination_criteria)

    height, width = frame.shape[:2]
    # stabilize the frame by applying the homographic matrix
    stabilized_frame = cv2.warpPerspective(
        frame, warp_matrix, (width, height), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    return stabilized_frame


def subtract_background(stabilized_frame, fgbg):

    # convert the stabilized frame to grayscale
    gray_frame = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY)

    # subtract background of grayscale frame
    fgmask = fgbg.apply(gray_frame)

    return fgmask


def morphological_segmentation(fgmask, kernel):
    # apply morphological opening (erosion and dilatation) on the foregorund mask
    new_fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # apply canny edge detection on the foreground mask
    edges = cv2.Canny(new_fgmask, 50, 190, 3)

    # thresholding resulting image to keep only edges
    _, new_fgmask = cv2.threshold(edges, 127, 255, 0)

    return new_fgmask


def circularity_filter(contour):
    # compute the area of the contour
    area = cv2.contourArea(contour)

    # compute the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)

    circularity = 0

    # check if perimeter is not null
    if perimeter != 0:

        # compute the ciruclarity of the contour
        circularity = 4*math.pi*area/perimeter**2

    return circularity >= 0.3 and circularity <= 0.8


def eccentricity_filter(contour):
    # extract the bounding rectangle coordinates and size
    x, y, w, h = cv2.boundingRect(contour)

    eccentricity = 1

    # check that the width of the blob is not null
    if w != 0:
        # compute eccentricity
        eccentricity = h/w

    return eccentricity <= 0.7


def size_filter(contour, min_area, max_area):

    area = cv2.contourArea(contour)

    return area > min_area and area < max_area


def color_filter(contour, ball_color_list):
    """Assert if the ball is close enough of the color set identified.
    
    <param contour> output of cv2.findContours
    <param color_list> list of strings containing one or more of these elements: white red green blue yellow black
    """
    return True


def lucas_kanade(img1, img2, contour, window_size=5):
    """Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        contour - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for [[y, x]] in contour:
        # Keypoints can be located between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        A1 = Ix[y-w: y+w+1, x-w: x+w+1]
        A2 = Iy[y-w: y+w+1, x-w: x+w+1]
        A = np.c_[A1.reshape(-1,1), A2.reshape(-1,1)]
        b = -It[y-w: y+w+1, x-w: x+w+1].reshape(-1,1)
        try:
            d = np.dot(np.linalg.inv(A.T.dot(A)), A.T.dot(b))
            flow_vectors.append(d.flatten())
        except Exception as err:
            flow_vectors.append([0,0])

    flow_vectors = np.array(flow_vectors)

    return flow_vectors


def compute_mean_norm(vector_list):
    norm_sum = 0
    for vector in vector_list:
        norm_sum += math.sqrt(vector[0]**2 + vector[1]**2)
    return norm_sum / len(vector_list)


def optic_flow_norm(contours, frame1, frame2):
    """Return the norm of the optic flow vector of one or many keypoints between 2 consecutive frames.
    
    <param frame1> oldest frame
    <param frame2> newest frame
    """
    of_norms = []
    for contour in contours:
        of = lucas_kanade(frame1, frame2, contour)
        of_norm = compute_mean_norm(of)
        of_norms.append(of_norm)
    return of_norms


def optic_flow_scores(contours, frame1, frame2, importance_factor=20):
    of_norms = optic_flow_norm(contours, frame1, frame2)
    of_scores = [of_norm * importance_factor for of_norm in of_norms]
    return of_scores


def is_head(contour):
    """Return a boolean indicating whether the given contours stands for a player head or not."""
    return False


def mean_y_pos(contour):
    """Return the mean y cordinate of the contour.
    A contour is an array of points with format [[x,y]].
    """
    mean = 0
    for point in contour:
        mean += point[0,1] # because point has format [[x,y]]
    return mean / len(contour)
        