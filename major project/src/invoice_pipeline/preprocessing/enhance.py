import cv2
import numpy as np


def enhance_image(image, is_digital=False):
    """
    Applies image preprocessing for better OCR accuracy:
    - Grayscale conversion
    - CLAHE contrast enhancement
    - Fast Non-Local Means noise reduction
    - Deskewing (auto-rotate if skewed)
    """
    # Upscale low-resolution images (phone photos, WhatsApp images) to standard document resolution
    h, w = image.shape[:2]
    if w < 1500:
        scale = max(2.0, 1800.0 / w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Noise reduction
    # Lighter denoising for digital PDFs to preserve decimal points
    denoise_strength = 3 if is_digital else 10
    denoised = cv2.fastNlMeansDenoising(enhanced, h=denoise_strength)

    # Auto-deskew
    angle = get_skew_angle(denoised)
    if abs(angle) > 0.5 and abs(angle) < 45.0:
        denoised = rotate_image(denoised, angle)

    return denoised


def get_skew_angle(gray):
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in contours:
        min_rect = cv2.minAreaRect(c)
        angle = min_rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.1 and abs(angle) < 45:
            angles.append(angle)

    return float(np.median(angles)) if len(angles) > 0 else 0.0


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
