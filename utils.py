import cv2


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
    edges = cv2.Canny(new_fgmask, 50, 190)

    # thresholding resulting image to keep only edges
    _, new_fgmask = cv2.threshold(edges, 127, 255, 0)

    return fgmask
