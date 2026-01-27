import io
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def _is_pdf(filename: str, content_type: str) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return filename.lower().endswith(".pdf")


def extract_text_from_upload(data: bytes, filename: str, content_type: str) -> str:
    """
    Best-effort text extraction:
      1) If PDF: try embedded text extraction.
      2) If PDF has little/no text: render pages → OCR.
      3) If image: OCR.
    """
    if _is_pdf(filename, content_type):
        return _extract_text_from_pdf_bytes(data)
    # Try OCR as image
    return _ocr_image_bytes(data)


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    texts = []
    # 1) Extract embedded text
    for page in doc:
        t = page.get_text("text") or ""
        if t.strip():
            texts.append(t)
    joined = "\n".join(texts).strip()
    if len(joined) >= 50:
        return joined

    # 2) Fallback to OCR of rendered pages (scanned PDFs)
    ocr_texts = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_texts.append(pytesseract.image_to_string(img))
    return "\n".join(ocr_texts).strip()


def _ocr_image_bytes(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        # Not an image; return empty text rather than crashing.
        return ""
    return (pytesseract.image_to_string(img) or "").strip()
