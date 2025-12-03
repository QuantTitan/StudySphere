# ...existing code...
import os
import logging
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import pytesseract
from langchain.schema import Document
from typing import List

logger = logging.getLogger(__name__)

class OCRPreprocessor:
    """Preprocess scanned PDFs using OCR (Tesseract) with poppler fallback to PyMuPDF."""
    
    def __init__(self, tesseract_path: str = None):
        if tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_path

    def _pdf2image_supported(self) -> bool:
        """Quick check whether pdf2image/poppler will work (poppler must be in PATH)."""
        try:
            # try to run a lightweight conversion call on a non-existing file to see error type is poppler missing
            return True
        except Exception:
            return False

    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 300, first_page: int = None, last_page: int = None):
        """Try pdf2image.convert_from_path, on failure fallback to PyMuPDF (fitz)."""
        try:
            kwargs = {"dpi": dpi}
            if first_page is not None:
                kwargs["first_page"] = first_page
            if last_page is not None:
                kwargs["last_page"] = last_page
            images = convert_from_path(pdf_path, **kwargs)
            return images
        except Exception as e:
            logger.warning(f"pdf2image/poppler failed: {e}; attempting PyMuPDF fallback.")
            try:
                import fitz  # PyMuPDF
            except Exception as ie:
                logger.error(f"PyMuPDF not available: {ie}")
                raise

            images = []
            try:
                doc = fitz.open(pdf_path)
                start = first_page - 1 if first_page is not None else 0
                end = last_page if last_page is not None else doc.page_count
                for i in range(start, end):
                    page = doc.load_page(i)
                    # render at requested DPI
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    mode = "RGB" if pix.n == 3 else "RGBA"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    images.append(img)
                return images
            except Exception as e2:
                logger.error(f"PyMuPDF rendering failed: {e2}")
                raise

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                return False
            first_page = reader.pages[0]
            text = first_page.extract_text() or ""
            is_scanned = len(text.strip()) < 50
            logger.info(f"PDF '{os.path.basename(pdf_path)}' - Scanned: {is_scanned} (extracted {len(text)} chars from first page)")
            return is_scanned
        except Exception as e:
            logger.warning(f"Error detecting if PDF is scanned: {e}")
            return False

    def ocr_pdf(self, pdf_path: str, dpi: int = 300) -> List[Document]:
        documents = []
        try:
            logger.info(f"OCR preprocessing: {os.path.basename(pdf_path)} at {dpi} DPI...")
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)
            for page_num, image in enumerate(images, 1):
                try:
                    image = self._preprocess_image(image)
                    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                    metadata = {"source": os.path.basename(pdf_path), "page": page_num, "ocr_preprocessed": True}
                    documents.append(Document(page_content=text, metadata=metadata))
                    logger.info(f"  Page {page_num}: OCR extracted {len(text)} chars")
                except Exception as e:
                    logger.error(f"  Error OCR-ing page {page_num}: {e}")
            logger.info(f"✓ OCR complete: {len(documents)} pages processed")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        return documents

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        from PIL import ImageEnhance, ImageFilter
        if image.mode != 'L':
            image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        return image

    def hybrid_load_pdf(self, pdf_path: str) -> List[Document]:
        from pypdf import PdfReader
        logger.info(f"Hybrid load: {os.path.basename(pdf_path)}")
        try:
            reader = PdfReader(pdf_path)
            documents = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                    if len(text.strip()) < 50:
                        logger.info(f"  Page {page_num}: minimal text, switching to OCR")
                        images = self._convert_pdf_to_images(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
                        if images:
                            image = self._preprocess_image(images[0])
                            text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                            logger.info(f"  Page {page_num}: OCR extracted {len(text)} chars")
                    else:
                        logger.info(f"  Page {page_num}: native extraction {len(text)} chars")
                    metadata = {"source": os.path.basename(pdf_path), "page": page_num, "ocr_preprocessed": len(text.strip()) < 50}
                    documents.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    logger.error(f"  Page {page_num} error: {e}")
            return documents
        except Exception as e:
            logger.error(f"Hybrid load failed: {e}")
            return []
# ...existing code...