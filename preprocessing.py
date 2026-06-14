from PIL import Image, ImageEnhance
import cv2
import numpy as np

def detect_screen(image):
    """
    Detect the 3DS screen in a smartphone photo using contour detection.

    Converts the image to grayscale, applies Canny edge detection, then finds
    the largest quadrilateral contour — which should correspond to the 3DS screen frame.

    Args:
        image (PIL.Image.Image): Raw smartphone photo.

    Returns:
        numpy.ndarray: Array of shape (4, 2) with the screen's corner coordinates,
                       or None if no quadrilateral was found.
    """
    img_np   = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect edges — strong pixel gradients mark the screen borders
    edges    = cv2.Canny(img_gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort largest to smallest — the 3DS screen is the biggest rectangle in the photo
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        # Simplify contour shape — tolerance = 2% of perimeter
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2) #numpy array of corners

    return None  # no quadrilateral found — caller should use a fixed-crop fallback


def order_corners(corners):
    """
    Sort 4 corner points into a consistent order: top-left, top-right, bottom-right, bottom-left.

    OpenCV returns contour points in arbitrary order. This function reorders them
    so that perspective transform math works correctly.

    The sorting relies on two geometric properties:
      - top-left has the smallest sum of (x + y)
      - bottom-right has the largest sum of (x + y)
      - top-right has the smallest difference of (x - y)
      - bottom-left has the largest difference of (x - y)

    Args:
        corners (numpy.ndarray): Array of shape (4, 2) with unordered corner coordinates.

    Returns:
        numpy.ndarray: Array of shape (4, 2), float32, ordered as [TL, TR, BR, BL].
    """
    ordered = np.zeros((4, 2), dtype="float32")

    s = corners.sum(axis=1)             # x + y for each point
    ordered[0] = corners[np.argmin(s)]  # top-left     — smallest x+y
    ordered[2] = corners[np.argmax(s)]  # bottom-right — largest  x+y

    d = np.diff(corners, axis=1)        # x - y for each point
    ordered[1] = corners[np.argmin(d)]  # top-right    — smallest x-y
    ordered[3] = corners[np.argmax(d)]  # bottom-left  — largest  x-y

    return ordered


def correct_perspective(image, corners):
    """
    Apply a perspective transform to produce a flat, front-facing view of the 3DS screen.

    Takes the 4 detected corners (possibly skewed from a camera angle) and maps them
    onto a straight rectangle using a homography matrix. The output dimensions are
    derived from the detected screen size to preserve the original aspect ratio.

    Args:
        image (PIL.Image.Image): Raw smartphone photo.
        corners (numpy.ndarray): Array of shape (4, 2) from detect_screen().

    Returns:
        PIL.Image.Image: Perspective-corrected image of the screen.
    """
    ordered = order_corners(corners)

    # Measure the screen's real dimensions from the detected corners
    targeted_width  = np.linalg.norm(ordered[0] - ordered[1])  # top-left → top-right (top edge)
    targeted_height = np.linalg.norm(ordered[0] - ordered[3])  # top-left → bottom-left (left edge)

    # Define the destination rectangle: a flat [0,0]→[w,h] rectangle
    dst_pts = np.array(
        [[0, 0], [targeted_width, 0], [targeted_width, targeted_height], [0, targeted_height]],
        dtype="float32"
    )

    # Compute the homography matrix (3×3) that maps src corners → dst rectangle
    M = cv2.getPerspectiveTransform(ordered, dst_pts)

    # Apply the transform — OpenCV needs a numpy array, not PIL
    img_np = np.array(image)
    warped = cv2.warpPerspective(img_np, M, (int(targeted_width), int(targeted_height)))

    return Image.fromarray(warped)

def crop_dialogue(image, ratio=0.30):
    # return PIL image croped on dialogue box
    pass

def enhance_contrast(image):
    # return PIL image with high contrast & cleaner
    pass

def super_resolve(image, scale=2):
    # returns ML upscaled image
    pass

def preprocess(image_path, debug=False):
    # full pipeline - chains 5 functions together
    pass
