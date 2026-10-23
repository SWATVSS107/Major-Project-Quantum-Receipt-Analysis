import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import cv2
import numpy as np

_paddle_instance = None


def get_paddle_ocr():
    global _paddle_instance
    if _paddle_instance is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_instance = PaddleOCR(use_angle_cls=True, lang='en', det_limit_side_len=2500, show_log=False)
            print("[OCR] PaddleOCR initialized successfully with det_limit_side_len=2500.")
        except Exception as e:
            print(f"[OCR] Warning: PaddleOCR failed to initialize: {e}. Falling back to Tesseract.")
            _paddle_instance = False
    return _paddle_instance


def extract_ocr_tokens(image, native_text=""):
    """
    Extracts text tokens with bounding boxes using PaddleOCR (falls back to Tesseract).
    Returns list of dicts: [{"text": str, "bbox": [x0, y0, x1, y1], "confidence": float}]
    """
    tokens = []

    ocr_engine = get_paddle_ocr()
    if ocr_engine:
        try:
            results = ocr_engine.ocr(image, cls=True)
            if results and results[0]:
                for res in results[0]:
                    bbox_coords = res[0]
                    text, conf = res[1]

                    xs = [pt[0] for pt in bbox_coords]
                    ys = [pt[1] for pt in bbox_coords]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

                    tokens.append({
                        "text": text.strip(),
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "confidence": float(conf)
                    })
                return tokens
        except Exception as e:
            print(f"[OCR] PaddleOCR processing error: {e}")

    # Fallback: Tesseract
    try:
        import pytesseract
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            txt = data['text'][i].strip()
            conf = float(data['conf'][i])
            if txt and conf > 0:
                x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                tokens.append({
                    "text": txt,
                    "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                    "confidence": conf / 100.0
                })
    except Exception as e:
        print(f"[OCR] Tesseract fallback error: {e}")

    return tokens
