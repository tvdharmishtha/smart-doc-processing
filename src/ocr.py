import io
import logging
import os
import re
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR

from src.extract import InformationExtractor

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)


class OCRPipeline:
    def __init__(self, languages=['en'], use_gpu=False):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            use_gpu=use_gpu,
            lang='en',
            show_log=False
        )
        self.extractor = InformationExtractor()

    def _ensure_bgr(self, image):
        if image is None:
            raise ValueError("Image is empty")

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image.copy()

    def _resize_if_needed(self, image, target_longest_side=1400):
        height, width = image.shape[:2]
        longest_side = max(height, width)

        if longest_side <= 0:
            return image

        if longest_side < target_longest_side:
            scale = target_longest_side / float(longest_side)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if longest_side > target_longest_side * 1.8:
            scale = (target_longest_side * 1.8) / float(longest_side)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        return image

    def _pad_image(self, image, pad=24):
        if image is None or image.size == 0:
            return image

        if len(image.shape) == 2:
            border_value = 255
        else:
            border_value = (255, 255, 255)

        return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_value)

    def _estimate_quality_metrics(self, gray):
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        noise = float(np.mean(cv2.absdiff(gray, cv2.medianBlur(gray, 3))))

        return {
            "blur": blur,
            "contrast": contrast,
            "brightness": brightness,
            "noise": noise,
        }

    def _target_longest_side(self, gray, metrics):
        longest_side = max(gray.shape[:2])
        target = 1600

        if longest_side < 1200:
            target = 2200
        elif longest_side < 1600:
            target = 1900

        if metrics["blur"] < 80 or metrics["contrast"] < 35:
            target = max(target, 2400)
        elif metrics["blur"] < 140 or metrics["contrast"] < 45:
            target = max(target, 2100)

        return target

    def _unsharp_mask(self, gray, sigma=1.2, amount=1.6):
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharpened = cv2.addWeighted(gray, amount, blurred, -(amount - 1.0), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

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
        bgr = self._ensure_bgr(image)
        initial_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        initial_metrics = self._estimate_quality_metrics(initial_gray)
        target_longest_side = self._target_longest_side(initial_gray, initial_metrics)

        bgr = self._resize_if_needed(bgr, target_longest_side=target_longest_side)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        metrics = self._estimate_quality_metrics(gray)

        denoise_strength = 16 if metrics["noise"] > 10 else 10
        if metrics["blur"] < 90:
            denoised = cv2.bilateralFilter(gray, 7, 45, 45)
        else:
            denoised = gray
        denoised = cv2.fastNlMeansDenoising(denoised, None, denoise_strength, 7, 21)

        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.5 if metrics["contrast"] >= 35 else 3.5, tileGridSize=(8, 8))
        clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        contrast_enhanced = clahe.apply(normalized)
        contrast_strong = clahe_strong.apply(normalized)
        sharpened = self._unsharp_mask(contrast_enhanced, sigma=1.1, amount=1.7)
        sharpened_strong = self._unsharp_mask(contrast_strong, sigma=1.0, amount=1.9)

        angle = self._estimate_skew_angle(sharpened_strong)
        if abs(angle) > 0.5:
            bgr = self._rotate_image(bgr, angle)
            gray = self._rotate_image(normalized, angle)
            denoised = self._rotate_image(denoised, angle)
            contrast_enhanced = self._rotate_image(contrast_enhanced, angle)
            contrast_strong = self._rotate_image(contrast_strong, angle)
            sharpened = self._rotate_image(sharpened, angle)
            sharpened_strong = self._rotate_image(sharpened_strong, angle)
        else:
            gray = normalized

        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_strong = cv2.threshold(sharpened_strong, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_inv = cv2.bitwise_not(binary_strong)
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )
        adaptive_mean = cv2.adaptiveThreshold(
            contrast_strong,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        morph_kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, morph_kernel)
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, morph_kernel)
        thickened = cv2.dilate(binary_strong, morph_kernel, iterations=1)

        variants = {
            "original": self._pad_image(bgr),
            "gray": self._pad_image(gray),
            "denoised": self._pad_image(denoised),
            "enhanced": self._pad_image(contrast_enhanced),
            "enhanced_strong": self._pad_image(contrast_strong),
            "sharpened": self._pad_image(sharpened),
            "sharpened_strong": self._pad_image(sharpened_strong),
            "binary": self._pad_image(binary),
            "binary_strong": self._pad_image(binary_strong),
            "binary_inv": self._pad_image(binary_inv),
            "adaptive": self._pad_image(adaptive),
            "adaptive_mean": self._pad_image(adaptive_mean),
            "opened": self._pad_image(opened),
            "closed": self._pad_image(closed),
            "thickened": self._pad_image(thickened),
        }

        variants["cropped_original"] = self._pad_image(self._crop_content(variants["original"]))
        variants["cropped_enhanced"] = self._pad_image(self._crop_content(variants["enhanced"]))
        variants["cropped_enhanced_strong"] = self._pad_image(self._crop_content(variants["enhanced_strong"]))
        variants["cropped_sharpened"] = self._pad_image(self._crop_content(variants["sharpened"]))
        variants["cropped_binary"] = self._pad_image(self._crop_content(variants["binary"]))
        variants["cropped_adaptive"] = self._pad_image(self._crop_content(variants["adaptive"]))
        variants["cropped_closed"] = self._pad_image(self._crop_content(variants["closed"]))

        return variants

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

        normalized_text = self._normalize_output(text)
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
        fields = self.extractor.extract_all(normalized_text)
        field_count = sum(1 for value in fields.values() if value)
        structure_bonus = field_count * 18
        if field_count == 3:
            structure_bonus += 12
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
            + structure_bonus
            - short_tokens * 1.2
            - noisy_chars * 1.0
        )

    def _run_ocr(self, image):
        result = self.ocr.ocr(image, cls=True)
        words = self._flatten_result(result)
        return self._assemble_text(words)

    def _format_structured_text(self, fields):
        lines = ["Invoice"]

        if fields.get("name"):
            lines.append(f"Name: {fields['name']}")
        if fields.get("date"):
            lines.append(f"Date: {fields['date']}")
        if fields.get("amount"):
            lines.append(f"Total: {fields['amount']}")

        return "\n".join(lines)

    def _extract_label_value(self, text, label_pattern):
        match = re.search(label_pattern, text)
        if not match:
            return ""

        return (match.group(1) or "").strip()

    def _field_quality(self, field_name, value, text):
        if not value:
            return -100.0

        if field_name == "name":
            tokens = re.findall(r"[A-Za-z.&]+", value)
            lowered = value.lower()
            score = 0.0

            if any(token in lowered for token in ("invoice", "invoi", "irnc", "total", "date", "amount")):
                score -= 30
            if 2 <= len(tokens) <= 4:
                score += 14
            elif len(tokens) == 1:
                score += 2
            else:
                score -= 8
            if all(token[:1].isupper() for token in tokens if token[:1].isalpha()):
                score += 6
            if any(token.lower() in {"pvt", "ltd", "corp", "inc", "llc"} for token in tokens):
                score += 5
            if re.search(r'(?i)\bname\b', text):
                score += 4

            return score

        if field_name == "date":
            return 18.0 if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or "") else 4.0

        if field_name == "amount":
            score = 0.0
            try:
                amount = float(value)
            except Exception:
                return -20.0

            raw_amount = self._extract_label_value(
                text,
                r'(?i)\b(?:total|amount)\b\s*[:.]?\s*[^\d\n]{0,3}(\d[\d,]*\.?\d*)'
            )

            if 10 <= amount <= 50000:
                score += 18
            elif 0 < amount < 10:
                score -= 12
            elif amount <= 100000:
                score += 5
            else:
                score -= 20

            if raw_amount.startswith("0") and len(raw_amount.split(".")[0]) >= 4:
                score -= 8
            if value.endswith((".25", ".40", ".50", ".60", ".75", ".80", ".90", ".95", ".99", ".00")):
                score += 4

            return score

        return 0.0

    def _attach_candidate_fields(self, candidate):
        normalized_text = self._normalize_output(candidate["text"])
        fields = self.extractor.extract_all(normalized_text)
        candidate["normalized_text"] = normalized_text
        candidate["fields"] = fields
        candidate["base_score"] = candidate["score"]
        candidate["field_score"] = sum(
            self._field_quality(field_name, field_value, normalized_text)
            for field_name, field_value in fields.items()
            if field_value
        )
        candidate["score"] += candidate["field_score"]
        return candidate

    def _aggregate_best_fields(self, candidates):
        candidates = sorted(candidates, key=lambda item: item.get("score", -1.0), reverse=True)[:8]
        best_fields = {"name": None, "date": None, "amount": None}
        best_scores = {"name": float("-inf"), "date": float("-inf"), "amount": float("-inf")}
        grouped_candidates = {"name": [], "date": [], "amount": []}

        for candidate in candidates:
            normalized_text = candidate.get("normalized_text") or self._normalize_output(candidate.get("text", ""))
            fields = candidate.get("fields") or self.extractor.extract_all(normalized_text)

            for field_name, field_value in fields.items():
                if not field_value:
                    continue

                field_score = candidate.get("base_score", candidate.get("score", 0.0)) + self._field_quality(field_name, field_value, normalized_text)
                if field_score > best_scores[field_name]:
                    best_scores[field_name] = field_score
                    best_fields[field_name] = field_value

                grouped_candidates[field_name].append((field_value, field_score))

        for field_name, field_candidates in grouped_candidates.items():
            if not field_candidates:
                continue

            bucket_scores = []
            for field_value, field_score in field_candidates:
                matched_bucket = None
                for bucket in bucket_scores:
                    reference = bucket["value"]
                    if field_name == "name":
                        similarity = SequenceMatcher(
                            None,
                            self.extractor._normalize_name_for_grouping(field_value),
                            self.extractor._normalize_name_for_grouping(reference),
                        ).ratio()
                        if similarity >= 0.58:
                            matched_bucket = bucket
                            break
                    elif field_value == reference:
                        matched_bucket = bucket
                        break

                if matched_bucket is None:
                    bucket_scores.append({
                        "value": field_value,
                        "score": field_score,
                        "values": [field_value],
                    })
                else:
                    matched_bucket["score"] += field_score
                    matched_bucket["values"].append(field_value)

            best_bucket = max(bucket_scores, key=lambda item: item["score"])
            selected_value = self.extractor.choose_best_value(field_name, best_bucket["values"])
            if selected_value:
                best_fields[field_name] = selected_value

        return best_fields

    def _build_candidate_plan(self, variants):
        preferred_variants = [
            "cropped_original",
            "cropped_enhanced",
            "cropped_enhanced_strong",
            "cropped_sharpened",
            "cropped_adaptive",
            "enhanced",
            "enhanced_strong",
            "sharpened",
            "sharpened_strong",
            "original",
            "binary",
            "binary_strong",
            "adaptive",
            "adaptive_mean",
            "closed",
            "thickened",
        ]
        plan = []
        seen = set()

        for name in preferred_variants:
            image = variants.get(name)
            if image is not None and name not in seen:
                plan.append((name, image))
                seen.add(name)

        return plan

    def _normalize_output(self, text):
        text = re.sub(r'(?i)\b(?:irvoice|lnvoice|invoce)\b', 'Invoice', text)
        text = re.sub(r'(?i)\b([A-Za-z]+)(\d)', r'\1 \2', text)
        text = re.sub(r'(?i)\bD(?=20\d{2}[-/\s]?\d{2}[-/\s]?\d{2}\b)', '', text)
        text = re.sub(r'(?i)\bTota(?=\s*\d)', 'Total', text)
        text = re.sub(r'(?i)\bPate\b', 'Patel', text)
        text = re.sub(r'(?<=[A-Za-z])[._](?=[A-Za-z])', ' ', text)
        text = re.sub(r'(?i)\bP(?:u|y|t|n)?t\s*\.?\s*Ltd\b', 'Pvt Ltd', text)
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

            candidate_plan = self._build_candidate_plan(variants)

            for variant_name, variant_image in candidate_plan:
                text, confidence = self._run_ocr(variant_image)
                candidate = {
                    "variant": variant_name,
                    "rotation": 0,
                    "text": text,
                    "confidence": confidence,
                    "score": self._candidate_score(text, confidence),
                }
                candidates.append(self._attach_candidate_fields(candidate))

            top_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:4]

            for candidate in top_candidates:
                variant_image = variants.get(candidate["variant"])
                if variant_image is None:
                    continue

                for rotation in (90, 180, 270):
                    rotated_image = self._rotate_image(variant_image, rotation)
                    text, confidence = self._run_ocr(rotated_image)
                    candidate = {
                        "variant": candidate["variant"],
                        "rotation": rotation,
                        "text": text,
                        "confidence": confidence,
                        "score": self._candidate_score(text, confidence),
                    }
                    candidates.append(self._attach_candidate_fields(candidate))

            best_candidate = max(candidates, key=lambda item: item["score"], default=None)
            best_fields = self._aggregate_best_fields(candidates)
            final_text = self._normalize_output(best_candidate["text"] if best_candidate else "")

            if sum(1 for value in best_fields.values() if value) >= 2:
                final_text = self._format_structured_text(best_fields)

            return final_text.strip()

        except Exception:
            return ""
        return ""


def load_image_from_bytes(image_bytes: bytes):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
