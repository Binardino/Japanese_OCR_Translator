from PIL import Image, ImageEnhance
import cv2
import numpy as np
from pathlib import Path

def detect_screen(image, debug=False):
    """
    Detect the 3DS screen in a smartphone photo using contour detection.

    Converts the image to grayscale, applies Canny edge detection, then finds
    the largest quadrilateral contour — which should correspond to the 3DS screen frame.

    Args:
        image (PIL.Image.Image): Raw smartphone photo.
        debug (bool): If True, saves debug_canny.jpg and prints contour info. Default: False.

    Returns:
        numpy.ndarray: Array of shape (4, 2) with the screen's corner coordinates,
                       or None if no quadrilateral was found.
    """
    img_np   = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect edges — strong pixel gradients mark the screen borders
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    edges    = cv2.Canny(img_blur, 80, 160)
    kernel   = np.ones((3, 3), np.uint8)
    edges    = cv2.dilate(edges, kernel, iterations=1)

    if debug:
        Image.fromarray(edges).save("debug_canny.jpg")

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort largest to smallest — the 3DS screen is the biggest rectangle in the photo
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if debug:
        print(f"Contours found: {len(contours)}")
        for i, contour in enumerate(contours[:5]):  # top 5 only
            hull = cv2.convexHull(contour) 
            approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
            print(f"  contour {i}: area={cv2.contourArea(contour):.0f}, sides={len(approx)}")



    min_area = 0.10 * img_gray.shape[0] * img_gray.shape[1]
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        # Simplify contour shape — tolerance = 2% of perimeter
        hull  = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2)  #numpy array of corners

    if debug:  # draw the largest contour for diagnosis when no quadrilateral was found
        debug_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 10)
        Image.fromarray(debug_img).save("debug_contours.jpg")

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
    """
    Crop the bottom portion of the screen where the dialogue box appears.

    On a perspective-corrected 3DS screen, the dialogue box is consistently
    located in the bottom ~30% of the image. This function isolates that zone
    for OCR processing.

    Note: only works reliably for dialogue screens. Menu screens or full-text
    screens (e.g. Pokémon team selection) are not supported by this approach.

    Args:
        image (PIL.Image.Image): Perspective-corrected screen image.
        ratio (float): Fraction of the image height to keep from the bottom. Default: 0.30.

    Returns:
        PIL.Image.Image: Cropped image containing only the dialogue box area.
    """
    width, height = image.size
    top = height - (ratio * height)  # start of the bottom ratio% zone

    return image.crop((0, int(top), width, height))

def enhance_contrast(image):
    """
    Improve text legibility using CLAHE contrast enhancement followed by sharpening.

    Two-step process:
      1. CLAHE (Contrast Limited Adaptive Histogram Equalization): improves contrast
         locally, tile by tile, which handles uneven lighting on the 3DS screen better
         than a global contrast adjustment. Operates on grayscale only.
      2. PIL Sharpness: adds edge crispness to the result, helping OCR distinguish
         character strokes from background.

    Args:
        image (PIL.Image.Image): Cropped dialogue box image.

    Returns:
        PIL.Image.Image: Contrast-enhanced and sharpened image, in RGB mode.
    """
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # CLAHE requires single-channel input
    enhanced = clahe.apply(img_gray)
    img_rgb  = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)          # back to RGB — PIL and manga-ocr expect 3 channels
    img_pil  = Image.fromarray(img_rgb)                            # numpy → PIL

    return ImageEnhance.Sharpness(img_pil).enhance(2.0)

def super_resolve(image, scale=2):
    """
    Upscale the image by a given factor using Lanczos resampling.

    PIL LANCZOS is a mathematical interpolation filter that produces sharper results
    than bilinear or bicubic for text-heavy images. Scale ×2 is sufficient to go from
    ~200px dialogue crops to ~400px, which helps manga-ocr recognize thin kanji strokes.

    Note: a learned super-resolution model (Real-ESRGAN, spandrel) would produce better
    results on low-quality console photos, but LANCZOS is a reliable and dependency-free fallback.

    Args:
        image (PIL.Image.Image): Input image (any size).
        scale (int): Upscaling factor. Default: 2.

    Returns:
        PIL.Image.Image: Upscaled image of size (w*scale, h*scale).
    """
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.LANCZOS)


def preprocess(image_path, debug=False):
    """
    Run the full preprocessing pipeline on a smartphone photo of a 3DS screen.

    Chains 5 stages in order:
      1. detect_screen()       — locate the 3DS screen frame via quadrilateral contour detection
      2. correct_perspective() — flatten the skewed camera angle into a frontal rectangle
      3. crop_dialogue()       — keep only the bottom 30% where the dialogue box appears
      4. enhance_contrast()    — CLAHE + sharpening to improve OCR readability
      5. super_resolve()       — ×2 Lanczos upscaling for better character recognition

    If detect_screen() fails (returns None), the raw photo is passed directly to
    crop_dialogue() as a fixed-crop fallback — no perspective correction applied.

    Args:
        image_path (str | Path): Path to the input smartphone photo.
        debug (bool): If True, saves intermediate images at each stage.
                      Files are named debug_<stem>_01_correct_perspective.jpg etc.

    Returns:
        PIL.Image.Image: Preprocessed image, ready for manga-ocr or Gemini Vision.
    """
    inputs = Image.open(image_path)
    stem   = Path(image_path).stem

    corners = detect_screen(image=inputs, debug=debug)
    if corners is not None:
        image    = correct_perspective(inputs, corners)
        if debug:
            image.save(f"debug_{stem}_01_correct_perspective.jpg")
    else:
        image = inputs  # fallback: no perspective correction

    dialogue = crop_dialogue(image)
    if debug:
        dialogue.save(f"debug_{stem}_02_crop_dialogue.jpg")

    enhanced = enhance_contrast(dialogue)
    if debug:
        enhanced.save(f"debug_{stem}_03_enhance_contrast.jpg")

    output   = super_resolve(enhanced)
    if debug:
        output.save(f"debug_{stem}_04_super_resolve.jpg")

    return output
