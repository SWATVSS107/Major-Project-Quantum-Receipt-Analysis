import os
import fitz  # PyMuPDF
import cv2
import numpy as np

def load_document(file_path):
    """
    Loads PDF or Image document and returns page representation:
    - Text lines (if vector PDF with native text)
    - Full image array (OpenCV format BGR) at 300 DPI for best OCR quality
    - Source type: 'pdf_text', 'pdf_scanned', or 'image'
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        doc = fitz.open(file_path)
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")
            pix = page.get_pixmap(dpi=300)  # 300 DPI for high-quality OCR
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

            has_native_text = len(text.strip()) > 50
            pages.append({
                "page_num": i + 1,
                "text": text,
                "image": img_np,
                "has_native_text": has_native_text,
                "width": pix.width,
                "height": pix.height
            })
        return {"type": "pdf", "pages": pages}
    else:
        # Image file: jpg, jpeg, png, tiff etc.
        img_np = cv2.imread(file_path)
        if img_np is None:
            raise ValueError(f"Could not load image file: {file_path}")
        h, w = img_np.shape[:2]
        return {
            "type": "image",
            "pages": [{
                "page_num": 1,
                "text": "",
                "image": img_np,
                "has_native_text": False,
                "width": w,
                "height": h
            }]
        }
