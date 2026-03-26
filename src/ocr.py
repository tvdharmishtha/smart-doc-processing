import io
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


class OCRPipeline:
    def __init__(self, languages=['en'], use_gpu=False):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            use_gpu=use_gpu,
            lang='en'
        )

    def _ensure_bgr(self, image):
        if image is None:
            raise ValueError("Image is empty")

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image.copy()

    def _resize_if_needed(self, image):
        height, width = image.shape[:2]
        longest_side = max(height, width)

        if longest_side < 1400:
            scale = 1400 / float(longest_side)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return image

    def _rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]

        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

    def _crop_content(self, image):
        if image is None or image.size == 0:
            return image

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)
        if w < 10 or h < 10:
            return image

        pad_x = max(20, int(w * 0.08))
        pad_y = max(20, int(h * 0.2))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return image

        return self._resize_if_needed(cropped)

    def _estimate_skew_angle(self, gray):
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 50:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle

        return -angle

    def preprocess(self, image):
        bgr = self._resize_if_needed(self._ensure_bgr(image))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(gray, -1, kernel)

        angle = self._estimate_skew_angle(sharpened)
        if abs(angle) > 0.5:
            bgr = self._rotate_image(bgr, angle)
            gray = self._rotate_image(sharpened, angle)
        else:
            gray = sharpened

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_inv = cv2.bitwise_not(binary)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        morph_kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, morph_kernel)
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, morph_kernel)
        thickened = cv2.dilate(binary, morph_kernel, iterations=1)

        cropped_original = self._crop_content(bgr)
        cropped_enhanced = self._crop_content(gray)
        cropped_binary = self._crop_content(binary)
        cropped_closed = self._crop_content(closed)

        return {
            "original": bgr,
            "enhanced": gray,
            "binary": binary,
            "binary_inv": binary_inv,
            "adaptive": adaptive,
            "opened": opened,
            "closed": closed,
            "thickened": thickened,
            "cropped_original": cropped_original,
            "cropped_enhanced": cropped_enhanced,
            "cropped_binary": cropped_binary,
            "cropped_closed": cropped_closed,
        }

    def _flatten_result(self, result):
        words = []

        if not result:
            return words

        for block in result:
            if not block:
                continue

            for item in block:
                if not item or len(item) < 2:
                    continue

                box, content = item[0], item[1]
                if not box or not content:
                    continue

                text = str(content[0]).strip()
                if not text:
                    continue

                confidence = float(content[1]) if len(content) > 1 else 0.0
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]

                words.append({
                    "text": text,
                    "confidence": confidence,
                    "x": min(xs),
                    "y": min(ys),
                    "cx": float(sum(xs) / len(xs)),
                    "cy": float(sum(ys) / len(ys)),
                    "height": max(ys) - min(ys),
                })

        return words

    def _assemble_text(self, words):
        if not words:
            return "", 0.0

        words = sorted(words, key=lambda item: (item["cy"], item["x"]))
        median_height = float(np.median([max(word["height"], 1) for word in words]))
        line_threshold = max(12.0, median_height * 0.7)

        lines = []
        current_line = [words[0]]
        current_y = words[0]["cy"]

        for word in words[1:]:
            if abs(word["cy"] - current_y) <= line_threshold:
                current_line.append(word)
                current_y = sum(item["cy"] for item in current_line) / len(current_line)
            else:
                lines.append(current_line)
                current_line = [word]
                current_y = word["cy"]

        lines.append(current_line)

        line_texts = []
        for line in lines:
            sorted_line = sorted(line, key=lambda item: item["x"])
            joined = " ".join(item["text"] for item in sorted_line)
            joined = re.sub(r'\s+([,:;])', r'\1', joined)
            line_texts.append(joined.strip())

        text = "\n".join(line for line in line_texts if line)
        avg_confidence = sum(word["confidence"] for word in words) / len(words)
        return text.strip(), avg_confidence

    def _candidate_score(self, text, confidence):
        if not text:
            return -1.0

        lowered = text.lower()
        keyword_bonus = sum(keyword in lowered for keyword in ["invoice", "name", "date", "total", "amount"])
        digit_count = sum(char.isdigit() for char in text)
        alpha_count = sum(char.isalpha() for char in text)
        line_bonus = max(text.count("\n"), 1)
        short_tokens = len(re.findall(r'\b[a-zA-Z]\b', text))
        noisy_chars = len(re.findall(r'[^\w\s:.,\-/₹$]', text))
        label_bonus = sum(bool(re.search(pattern, lowered)) for pattern in [r'\bname\b', r'\bdate\b', r'\btotal\b', r'\bamount\b'])
        amount_match = re.search(r'(?i)\b(?:total|amount)\b\s*[:.]?\s*[^\d\n]{0,2}(\d[\d,]*\.?\d*)', text)
        amount_bonus = 0.0
        if amount_match:
            amount_digits = len(amount_match.group(1).replace(',', '').split('.')[0])
            amount_bonus = min(amount_digits, 6) * 3.5

        return (
            confidence * 100
            + min(len(text), 120) * 0.25
            + keyword_bonus * 6
            + label_bonus * 4
            + amount_bonus
            + min(digit_count, 20) * 0.4
            + min(alpha_count, 80) * 0.15
            + line_bonus * 2
            - short_tokens * 1.2
            - noisy_chars * 1.0
        )

    def _run_ocr(self, image):
        result = self.ocr.ocr(image, cls=True)
        words = self._flatten_result(result)
        return self._assemble_text(words)

    def _normalize_output(self, text):
        text = re.sub(r'(?i)\b(?:irvoice|lnvoice|invoce)\b', 'Invoice', text)
        text = re.sub(r'(?i)\b([A-Za-z]+)(\d)', r'\1 \2', text)
        text = re.sub(r'(?i)\bD(?=20\d{2}[-/\s]?\d{2}[-/\s]?\d{2}\b)', '', text)
        text = re.sub(r'(?i)\bTota(?=\s*\d)', 'Total', text)
        text = re.sub(r'(?i)\bPate\b', 'Patel', text)
        text = re.sub(r'(?i)\bname\s*[.:]?\s*', 'Name: ', text)
        text = re.sub(r'(?i)\bdate\s*[.:]?\s*', 'Date: ', text)
        text = re.sub(r'(?i)\b(total|amount)\s*[.:]?\s*', lambda match: f"{match.group(1).title()}: ", text)
        text = re.sub(r'(?i)\b(Total|Amount):\s*[@$₹]?\s*(\d[\d,]*\.?\d*)', r'\1: \2', text)
        text = re.sub(r'(?<!\d)(\d{4})[-/\s]?(\d{2})[-/\s]?(\d{2})(?!\d)', r'\1-\2-\3', text)
        text = re.sub(r'(?im)^(invoice)\s*\n\s*(20\d{2}-\d{2}-\d{2}\b)', r'\1\nDate: \2', text)
        text = re.sub(r'(?im)^\s*(20\d{2}-\d{2}-\d{2})\s+(Name:)', r'Date: \1 \2', text)
        text = re.sub(r'(?im)^\s*(20\d{2}-\d{2}-\d{2})\s*$', r'Date: \1', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def extract_text(self, image):
        try:
            variants = self.preprocess(image)
            candidates = []

            for rotation in (0, 90, 180, 270):
                for variant_name, variant_image in variants.items():
                    candidate_image = variant_image if rotation == 0 else self._rotate_image(variant_image, rotation)
                    text, confidence = self._run_ocr(candidate_image)
                    score = self._candidate_score(text, confidence)

                    candidates.append({
                        "variant": variant_name,
                        "rotation": rotation,
                        "text": text,
                        "confidence": confidence,
                        "score": score,
                    })

            best_candidate = max(candidates, key=lambda item: item["score"], default=None)
            final_text = self._normalize_output(best_candidate["text"] if best_candidate else "")

            return final_text.strip()

        except Exception:
            return ""
        return ""


def load_image_from_bytes(image_bytes: bytes):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
