import os
import cv2
import json  # NEW
import shutil  # NEW
import numpy as np
import base64
import re
import logging
import subprocess  # NEW
from tempfile import NamedTemporaryFile  # NEW
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional

# OCR imports with fallback handling
OCR_ENGINES = []
try:
    import pytesseract
    OCR_ENGINES.append('tesseract')
    logging.info("Tesseract OCR available")
except ImportError:
    logging.warning("Tesseract not available")

try:
    import pytesseract
    OCR_ENGINES.append('easyocr')
    logging.info("EasyOCR available")
except ImportError:
    logging.warning("EasyOCR not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReceiptOCR:
    """Advanced Receipt OCR processor with LLM-first (receipt-ocr CLI) + dual-engine fallback"""

    def __init__(self):
        self.primary_ocr = None
        self.backup_ocr = None
        self.easyocr_reader = None

        # receipt-ocr CLI presence (LLM-backed parser)
        self.receipt_ocr_cli = shutil.which("receipt-ocr")  # NEW
        if self.receipt_ocr_cli:
            logger.info(f"'receipt-ocr' CLI found at: {self.receipt_ocr_cli}")  # NEW
        else:
            logger.info("'receipt-ocr' CLI not found—will use EasyOCR/Tesseract fallback")  # NEW

        # Initialize OCR engines
        self._initialize_ocr_engines()

        # Receipt parsing patterns
        self._compile_patterns()

        logger.info(f"ReceiptOCR initialized with engines: {OCR_ENGINES}")

    def _initialize_ocr_engines(self):
        """Initialize available OCR engines"""
        if 'easyocr' in OCR_ENGINES:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.primary_ocr = 'easyocr'
                logger.info("EasyOCR initialized as primary engine")
            except Exception as e:
                logger.error(f"EasyOCR initialization failed: {e}")

        if 'tesseract' in OCR_ENGINES:
            try:
                pytesseract.get_tesseract_version()
                if not self.primary_ocr:
                    self.primary_ocr = 'tesseract'
                else:
                    self.backup_ocr = 'tesseract'
                logger.info(f"Tesseract available as {'primary' if self.primary_ocr == 'tesseract' else 'backup'} engine")
            except Exception as e:
                logger.error(f"Tesseract not available: {e}")

        if not self.primary_ocr and not self.receipt_ocr_cli:
            logger.error("No local OCR engines available! Install pytesseract/easyocr or the receipt-ocr CLI")
            raise RuntimeError("No OCR engines available")

    def _compile_patterns(self):
        """Compile regex patterns for receipt parsing"""
        self.patterns = {
            'store_name': [
                r'^([A-Z][A-Z\s&]+[A-Z])\s*$',
                r'^([A-Z][A-Za-z\s&\.]+(?:MARKET|STORE|SHOP|MART|FOODS))',
                r'^([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*$'
            ],
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})'
            ],
            'time': [
                r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)',
                r'(\d{2}:\d{2}:\d{2})'
            ],
            'total': [
                r'TOTAL[\s:]*\$?(\d+\.?\d*)',
                r'AMOUNT\s+DUE[\s:]*\$?(\d+\.?\d*)',
                r'BALANCE[\s:]*\$?(\d+\.?\d*)'
            ],
            'tax': [
                r'TAX[\s:]*\$?(\d+\.?\d*)',
                r'HST[\s:]*\$?(\d+\.?\d*)',
                r'GST[\s:]*\$?(\d+\.?\d*)'
            ],
            'item': [
                r'([A-Za-z][A-Za-z\s\-/&0-9]+?)\s+(\d+\.?\d*)(?:\s*$)',
                r'([A-Za-z][A-Za-z\s\-/&0-9]+?)\s+\$(\d+\.?\d*)(?:\s*$)',
                r'^(\d+)\s+([A-Za-z][A-Za-z\s\-/&0-9]+?)\s+\$?(\d+\.?\d*)$',
                r'([A-Za-z][A-Za-z\s\-/&0-9]+?)\s+(\d+\.?\d*)\s*(?:LB|KG|OZ|G)?\s+\$?(\d+\.?\d*)$'
            ]
        }

    def process_receipt(self, base64_image: str) -> Dict:
        """Main receipt processing pipeline"""
        try:
            start_time = datetime.now()

            # Decode and preprocess image
            image = self._decode_base64_image(base64_image)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to decode image',
                    'timestamp': datetime.now().isoformat()
                }

            # === 1) Try LLM-backed 'receipt-ocr' CLI first ====================  # NEW
            if self.receipt_ocr_cli:
                llm_res = self._extract_with_receipt_ocr_cli(image)
                if llm_res.get('success'):
                    llm_res['processing_info']['processing_time'] = round(
                        (datetime.now() - start_time).total_seconds(), 2
                    )
                    return llm_res
                else:
                    logger.info(f"receipt-ocr CLI failed or returned no data: {llm_res.get('error')}")

            # === 2) Fallback to classic OCR ====================================
            processed_image = self._preprocess_image(image)
            ocr_results = self._extract_text_multi_engine(processed_image)

            if not ocr_results['text']:
                return {
                    'success': False,
                    'error': 'No text could be extracted from image',
                    'timestamp': datetime.now().isoformat()
                }

            # Parse structured data from extracted text
            structured_data = self._parse_receipt_data(ocr_results['text'])
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'raw_text': ocr_results['text'],
                'structured_data': structured_data,
                'processing_info': {
                    'primary_ocr': ocr_results['engine'],
                    'confidence': ocr_results['confidence'],
                    'processing_time': round(processing_time, 2),
                    'image_size': f"{image.shape[1]}x{image.shape[0]}",
                    'preprocessing_applied': True
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Receipt processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # ------------------- NEW: LLM/CLI path -------------------
    def _extract_with_receipt_ocr_cli(self, image: np.ndarray) -> Dict:
        """
        Call the 'receipt-ocr' CLI with a temp JPG and map its JSON to our schema.
        Expects env vars for provider (e.g. OPENAI_API_KEY + OPENAI_MODEL or base_url overrides).
        """
        try:
            # Write image to a temporary file as JPG
            with NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if not ok:
                    raise RuntimeError("Failed to encode image to JPEG for receipt-ocr")
                tmp.write(buf.tobytes())
                tmp.flush()

                # Run CLI
                proc = subprocess.run(
                    [self.receipt_ocr_cli, tmp.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )

            if proc.returncode != 0:
                return {
                    'success': False,
                    'error': f"receipt-ocr CLI error: {proc.stderr.strip() or proc.stdout[:200]}"
                }

            # CLI prints JSON on stdout (per README)
            payload = proc.stdout.strip()
            data = json.loads(payload)

            # Map CLI JSON (sample in README) to our schema
            items = []
            for it in data.get("line_items", []) or []:
                items.append({
                    "name": it.get("item_name", "").strip(),
                    "price": float(it.get("item_total", it.get("item_price", 0.0)) or 0.0),
                    "quantity": int(it.get("item_quantity", 1) or 1),
                })

            # Build structured schema aligned with your parser keys
            structured = {
                "store_name": data.get("merchant_name", "") or data.get("merchant", ""),
                "date": data.get("transaction_date", "") or data.get("date", ""),
                "time": data.get("transaction_time", "") or data.get("time", ""),
                "total": float(data.get("total_amount", data.get("total", 0.0)) or 0.0),
                "tax": float(data.get("tax_amount", data.get("tax", 0.0)) or 0.0),
                "subtotal": float(data.get("subtotal_amount", data.get("subtotal", 0.0)) or 0.0),
                "items": items,
                "payment_method": data.get("payment_method", "")
            }

            # If subtotal missing but tax+total exist, compute it
            if not structured["subtotal"] and structured["total"] and structured["tax"]:
                structured["subtotal"] = round(structured["total"] - structured["tax"], 2)

            # Compose success result (we may not have "raw_text" from CLI)
            return {
                "success": True,
                "raw_text": "",  # CLI returns structured JSON; raw text not guaranteed
                "structured_data": structured,
                "processing_info": {
                    "primary_ocr": "receipt-ocr-cli",
                    "confidence": 0,  # CLI doesn't expose a confidence; keep 0 or derive later
                    "preprocessing_applied": False,
                    "image_size": f"{image.shape[1]}x{image.shape[0]}",
                },
                "timestamp": datetime.now().isoformat()
            }

        except json.JSONDecodeError as je:
            return {"success": False, "error": f"Failed to parse receipt-ocr JSON: {je}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------- Existing helpers -------------------
    def _decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            image_bytes = base64.b64decode(base64_string)
            pil_image = Image.open(BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            height, width = cleaned.shape
            if height < 600 or width < 400:
                scale = max(600/height, 400/width)
                cleaned = cv2.resize(
                    cleaned,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_CUBIC
                )
            return cleaned
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image

    def _extract_text_multi_engine(self, image: np.ndarray) -> Dict:
        results = {'text': '', 'confidence': 0, 'engine': 'none'}
        if self.primary_ocr == 'easyocr':
            results = self._extract_text_easyocr(image)
        elif self.primary_ocr == 'tesseract':
            results = self._extract_text_tesseract(image)

        if results['confidence'] < 70 and self.backup_ocr:
            logger.info(f"Primary OCR confidence low ({results['confidence']}), trying backup engine")
            backup_results = None
            if self.backup_ocr == 'tesseract':
                backup_results = self._extract_text_tesseract(image)
            elif self.backup_ocr == 'easyocr':
                backup_results = self._extract_text_easyocr(image)
            if backup_results and backup_results['confidence'] > results['confidence']:
                logger.info(f"Backup engine performed better ({backup_results['confidence']} vs {results['confidence']})")
                results = backup_results
        return results

    def _extract_text_tesseract(self, image: np.ndarray) -> Dict:
        try:
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,/$%-:'
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(image, config=config)
            confs = []
            for c in data.get('conf', []):
                try:
                    ci = int(float(c))
                    if ci > 0:
                        confs.append(ci)
                except Exception:
                    continue
            avg_conf = sum(confs) / len(confs) if confs else 0
            return {'text': text.strip(), 'confidence': round(avg_conf, 1), 'engine': 'tesseract'}
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'tesseract_failed'}

    def _extract_text_easyocr(self, image: np.ndarray) -> Dict:
        try:
            results = self.easyocr_reader.readtext(image)
            text_lines, confidences = [], []
            for (_bbox, text, conf) in results:
                if conf > 0.3:
                    text_lines.append(text)
                    confidences.append(conf * 100)
            return {
                'text': '\n'.join(text_lines),
                'confidence': round((sum(confidences) / len(confidences)) if confidences else 0, 1),
                'engine': 'easyocr'
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'easyocr_failed'}

    def _parse_receipt_data(self, text: str) -> Dict:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        structured_data = {
            'store_name': '',
            'date': '',
            'time': '',
            'total': 0.0,
            'tax': 0.0,
            'subtotal': 0.0,
            'items': [],
            'payment_method': ''
        }

        # Store name
        for line in lines[:5]:
            for pattern in self.patterns['store_name']:
                match = re.search(pattern, line.upper())
                if match and len(match.group(1)) > 3:
                    structured_data['store_name'] = match.group(1).title()
                    break
            if structured_data['store_name']:
                break

        # Date & time
        for line in lines:
            if not structured_data['date']:
                for pattern in self.patterns['date']:
                    m = re.search(pattern, line)
                    if m:
                        structured_data['date'] = m.group(1)
                        break
            if not structured_data['time']:
                for pattern in self.patterns['time']:
                    m = re.search(pattern, line)
                    if m:
                        structured_data['time'] = m.group(1)
                        break

        # Totals
        for line in lines:
            u = line.upper()
            for pattern in self.patterns['total']:
                m = re.search(pattern, u)
                if m:
                    try:
                        structured_data['total'] = float(m.group(1))
                        break
                    except ValueError:
                        pass
            for pattern in self.patterns['tax']:
                m = re.search(pattern, u)
                if m:
                    try:
                        structured_data['tax'] = float(m.group(1))
                        break
                    except ValueError:
                        pass

        # Items
        structured_data['items'] = self._extract_items(lines)

        # Subtotal
        if structured_data['total'] and structured_data['tax']:
            structured_data['subtotal'] = round(structured_data['total'] - structured_data['tax'], 2)

        return structured_data

    def _extract_items(self, lines: List[str]) -> List[Dict]:
        """Extract individual items from receipt lines (cleaned & de-duped)"""
        items = []
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            # Skip header/footer lines
            if any(line.upper().startswith(k) for k in
                   ['TOTAL', 'TAX', 'SUBTOTAL', 'CHANGE', 'CASH', 'CREDIT', 'DEBIT', 'THANK YOU']):
                continue

            for pattern in self.patterns['item']:
                m = re.search(pattern, line)
                if not m:
                    continue
                try:
                    groups = m.groups()
                    if len(groups) == 2:  # name, price
                        name, price = groups
                        items.append({'name': name.strip(), 'price': float(price), 'quantity': 1})
                    elif len(groups) == 3:
                        if groups[0].isdigit():  # qty, name, price
                            qty, name, price = groups
                            items.append({'name': name.strip(), 'price': float(price), 'quantity': int(qty)})
                        else:  # name, weight, price
                            name, weight, price = groups
                            items.append({'name': f"{name.strip()} ({weight})", 'price': float(price), 'quantity': 1})
                    break
                except (ValueError, IndexError):
                    continue
        return items
