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
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect edges — strong pixel gradients mark the screen borders
    edges = cv2.Canny(img_gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort largest to smallest — the 3DS screen is the biggest rectangle in the photo
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        # Simplify contour shape — tolerance = 2% of perimeter
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None  # no quadrilateral found — caller should use a fixed-crop fallback


def correct_perspective(image, corners):
    # returns PIL image corrected
    pass

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
