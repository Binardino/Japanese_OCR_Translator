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
    ordered = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]  # top-left
    ordered[2] = corners[np.argmax(s)]  # bottom-right
    d = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(d)]  # top-right
    ordered[3] = corners[np.argmax(d)]  # bottom-left
    return ordered


def correct_perspective(image, corners):
    # returns PIL image corrected
    orderned_corners = order_corners(corners)

    targeted_width  = np.linalg.norm(orderned_corners[0] - orderned_corners[1]) #compute top edge width   i.e [0] top-left -> [1] = top-right
    targeted_height = np.linalg.norm(orderned_corners[0] - orderned_corners[3]) #compute left edge height i.e [0] top-left -> [3] = bottom left

    dst_pts = np.array([[0,0], [targeted_width,0], [targeted_width,targeted_height], [0,targeted_height]], dtype="float32")
    M       = cv2.getPerspectiveTransform(orderned_corners, dst_pts)
    img_np  = np.array(image)
    warped  = cv2.warpPerspective(img_np, M, (int(targeted_width), int(targeted_height)))

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
