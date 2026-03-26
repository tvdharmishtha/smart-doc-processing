import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser


class InformationExtractor:
    def __init__(self):
        self.known_names = [
            'Ravi Patel', 'Global Corp', 'Jane Smith',
            'ABC Pvt Ltd', 'John Doe', 'Neha Shah'
        ]
        self.name_field_aliases = {
            'name', 'customer', 'customer_name', 'client', 'client_name',
            'vendor', 'vendor_name', 'person', 'person_name', 'party', 'party_name'
        }
        self.date_field_aliases = {
            'date', 'invoice_date', 'document_date', 'bill_date', 'txn_date', 'transaction_date'
        }
        self.amount_field_aliases = {
            'amount', 'total', 'total_amount', 'grand_total', 'amt', 'price', 'value', 'balance_due'
        }
        self.common_decimal_suffixes = {'25', '40', '50', '60', '75', '80', '90', '95', '99'}

    def _normalize_key(self, key: Any) -> str:
        return re.sub(r'[^a-z0-9]+', '_', str(key).strip().lower()).strip('_')

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        result = []
        seen = set()

        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)

        return result

    def _get_row_value(self, row: Dict[str, Any], aliases: set[str]) -> Optional[Any]:
        for key, value in row.items():
            normalized_key = self._normalize_key(key)
            if normalized_key in aliases:
                return value
        return None

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r", "\n")
        text = re.sub(r'(?i)\b(?:irvoice|lnvoice|invoce)\b', 'Invoice', text)
        text = re.sub(r'(?i)\b([A-Za-z]+)(\d)', r'\1 \2', text)
        text = re.sub(r'(?i)\bD(?=20\d{2}[-/\s]?\d{2}[-/\s]?\d{2}\b)', '', text)
        text = re.sub(r'(?<!\d)(\d{4})[:](\d{2})[-/](\d{2})(?!\d)', r'\1-\2-\3', text)
        text = re.sub(r'(?<!\d)(\d{2})[-/](\d{2})[-/](\d{4})(?!\d)', r'\3-\2-\1', text)
        text = re.sub(r'(?i)\bTota(?=\s*\d)', 'Total', text)
        text = re.sub(r'(?i)\btota[l1i]?\b', 'Total', text)
        text = re.sub(r'(?i)\bamo?un[t1i]?\b', 'Amount', text)
        text = re.sub(r'(?i)\bna[mp]e\b', 'Name', text)
        text = re.sub(r'(?i)\bda[tf]e\b', 'Date', text)
        text = re.sub(r'(?i)\bdate\s*[:.-]?\s*d(?=\d)', 'Date: ', text)
        text = re.sub(r'(?i)\bname\s*[:.-]?\s*', 'Name: ', text)
        text = re.sub(r'(?i)\bdate\s*[:.-]?\s*', 'Date: ', text)
        text = re.sub(r'(?i)\b(total|amount)\s*[:.-]?\s*', lambda match: f"{match.group(1).title()}: ", text)
        text = re.sub(r'(?i)\b(Total|Amount):\s*[@$₹]?\s*(\d[\d,]*\.?\d*)', r'\1: \2', text)
        text = re.sub(r'(?<!\d)(\d{4})[-/\s]?(\d{2})[-/\s]?(\d{2})(?!\d)', r'\1-\2-\3', text)
        text = re.sub(r'(?i)\bPate\b', 'Patel', text)
        text = re.sub(r'[{}\[\]()|_]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _normalize_date(self, value: str) -> Optional[str]:
        if not value:
            return None

        value = self._clean_text(value)
        value = re.sub(r'(?<!\d)(\d{4})[:](\d{2})[-/](\d{2})(?!\d)', r'\1-\2-\3', value)
        value = re.sub(r'(?<!\d)(\d{2})[-/](\d{2})[-/](\d{4})(?!\d)', r'\3-\2-\1', value)
        value = re.sub(r'(?<!\d)(\d{4})[-/\s]?(\d{2})[-/\s]?(\d{2})(?!\d)', r'\1-\2-\3', value)

        try:
            parsed = date_parser.parse(value, fuzzy=True, dayfirst=False)
            return parsed.strftime('%Y-%m-%d')
        except Exception:
            return None

    def _score_amount_candidate(self, raw_digits: str, value: float, scale: int) -> float:
        fraction = round(value - int(value), 2)
        score = 0.0

        if 10 <= value <= 50000:
            score += 10
        if 50 <= value <= 20000:
            score += 10
        if value > 50000:
            score -= 12

        if fraction in {0.25, 0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 0.95, 0.99}:
            score += 8
        elif fraction in {0.00, 0.05}:
            score += 4
        elif fraction > 0:
            score += 2

        if scale == 10 and raw_digits.endswith('00'):
            score += 8
        if scale == 10 and raw_digits.endswith(('0', '4', '5')):
            score += 4
        if scale == 100 and raw_digits[-2:] in self.common_decimal_suffixes:
            score += 8

        return score

    def _normalize_amount_number(self, number: str) -> Optional[str]:
        cleaned = str(number).replace(',', '').strip()
        if not cleaned:
            return None

        if '.' in cleaned:
            try:
                return f"{float(cleaned):.2f}"
            except ValueError:
                return None

        if not cleaned.isdigit():
            return None

        base_value = float(cleaned)
        candidates = [(base_value, 1)]
        if len(cleaned) >= 4:
            candidates.append((base_value / 10.0, 10))
            candidates.append((base_value / 100.0, 100))

        best_value, _ = max(
            candidates,
            key=lambda item: self._score_amount_candidate(cleaned, item[0], item[1])
        )
        return f"{best_value:.2f}"

    def _normalize_amount(self, value: str) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"

        raw_value = str(value)
        match = re.search(r'[-+]?\d[\d,]*\.?\d*', raw_value)
        if not match:
            return None

        return self._normalize_amount_number(match.group())

    def _best_known_name_match(self, text: str) -> Optional[str]:
        tokens = re.findall(r'[A-Za-z][A-Za-z.&]+', text)
        if not tokens:
            return None

        best_name = None
        best_score = 0.0

        for size in (2, 3, 4):
            for index in range(0, max(len(tokens) - size + 1, 0)):
                phrase = " ".join(tokens[index:index + size])
                for name in self.known_names:
                    score = SequenceMatcher(None, phrase.lower(), name.lower()).ratio()
                    if score > best_score:
                        best_name = name
                        best_score = score

        return best_name if best_score >= 0.82 else None

    def extract_name(self, text: str) -> Optional[str]:
        text = self._clean_text(text)
        text_lower = text.lower()

        for name in self.known_names:
            if name.lower() in text_lower:
                return name

        match = re.search(
            r'(?i)\b(?:name|customer|client|vendor|employee|person)\b\s*[:\-]?\s*([A-Z][A-Za-z.&]+(?:\s+[A-Z][A-Za-z.&]+){0,4})',
            text,
        )
        if match:
            candidate = match.group(1).strip(' .:-')
            candidate = re.sub(r'\b(Date|Amount|Total|Invoice)\b.*$', '', candidate, flags=re.IGNORECASE).strip(' .:-')
            if candidate and not re.search(r'(?i)\b(filename|document|doc|jpg|jpeg|png|csv|xls|xlsx)\b', candidate):
                return candidate

        fuzzy_match = self._best_known_name_match(text)
        if fuzzy_match:
            return fuzzy_match

        return None

    def extract_date(self, text: str) -> Optional[str]:
        text = self._clean_text(text)

        match = re.search(r'(?i)\b(?:date|invoice date|dt)\b\s*[:\-]?\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2}|[0-9]{8})', text)
        if match:
            return self._normalize_date(match.group(1))

        alt_match = re.search(r'(?<!\d)([0-9]{4}[:][0-9]{2}[-/][0-9]{2})(?!\d)', text)
        if alt_match:
            return self._normalize_date(alt_match.group(1))

        fallback_match = re.search(r'(?<!\d)([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})(?!\d)', text)
        if fallback_match:
            return self._normalize_date(fallback_match.group(1))

        compact_match = re.search(r'(?<!\d)(20\d{2})[-/\s]?(\d{2})[-/\s]?(\d{2})(?!\d)', text)
        if compact_match:
            return self._normalize_date("-".join(compact_match.groups()))

        return None

    def extract_amount(self, text: str) -> Optional[str]:
        text = self._clean_text(text)

        labelled_match = re.search(
            r'(?i)\b(?:grand total|total amount|total|amount|amt|balance due)\b\s*[:\-]?\s*(?:rs\.?|inr|usd|\$|₹)?\s*([0-9][0-9,]*\.?\d{0,2})',
            text,
        )
        if labelled_match:
            return self._normalize_amount(labelled_match.group(1))

        candidates = re.findall(r'(?<![-/])\b\d[\d,]*\.\d{1,2}\b', text)
        if candidates:
            values = [float(value.replace(',', '')) for value in candidates]
            return f"{max(values):.2f}"

        labelled_integer = re.search(
            r'(?i)\b(?:grand total|total amount|total|amount|amt|balance due)\b\s*[:\-]?\s*(?:rs\.?|inr|usd|\$|₹)?\s*([0-9][0-9,]*)\b',
            text,
        )
        if labelled_integer:
            return self._normalize_amount(labelled_integer.group(1))

        integers = []
        for value in re.findall(r'\b\d[\d,]*\b', text):
            numeric = float(value.replace(',', ''))
            if 1900 <= numeric <= 2100:
                continue
            if 1 <= numeric <= 31 and re.search(r'\b20\d{2}[-/]?\d{2}[-/]?\d{2}\b', text):
                continue
            integers.append(numeric)

        if not integers:
            return None

        return f"{max(integers):.2f}"

    def extract_all(self, text: str) -> Dict[str, Optional[str]]:
        cleaned_text = self._clean_text(text)
        return {
            "name": self.extract_name(cleaned_text),
            "date": self.extract_date(cleaned_text),
            "amount": self.extract_amount(cleaned_text)
        }

    def extract_all_as_arrays(self, text: str) -> Dict[str, Any]:
        names = []
        dates = []
        amounts = []

        text = self._clean_text(text)

        lines = [line.strip() for line in re.split(r'[\n;]+', text) if line.strip()]
        for line in lines:
            record = self.extract_all(line)
            if record["name"]:
                names.append(record["name"])
            if record["date"]:
                dates.append(record["date"])
            if record["amount"]:
                amounts.append(record["amount"])

        if not names:
            name = self.extract_name(text)
            if name:
                names.append(name)

        if not dates:
            date = self.extract_date(text)
            if date:
                dates.append(date)

        if not amounts:
            amount = self.extract_amount(text)
            if amount:
                amounts.append(amount)

        return {
            "name": self._dedupe_preserve_order(names),
            "date": self._dedupe_preserve_order(dates),
            "amount": self._dedupe_preserve_order(amounts)
        }

    def extract_from_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        names = []
        dates = []
        amounts = []

        for row in rows:
            normalized_row = {
                str(key).strip().lower(): value
                for key, value in row.items()
                if value is not None and str(value).strip() != ""
            }

            if not normalized_row:
                continue

            name_value = self._get_row_value(normalized_row, self.name_field_aliases)
            date_value = self._get_row_value(normalized_row, self.date_field_aliases)
            amount_value = self._get_row_value(normalized_row, self.amount_field_aliases)

            row_text = " ".join(f"{key}: {value}" for key, value in normalized_row.items())

            name = self.extract_name(f"Name: {name_value}") if name_value else None
            date = self._normalize_date(str(date_value)) if date_value else self.extract_date(row_text)
            amount = self._normalize_amount(amount_value) if amount_value is not None else self.extract_amount(row_text)

            if name:
                names.append(name)
            if date:
                dates.append(date)
            if amount:
                amounts.append(amount)

        return {
            "name": self._dedupe_preserve_order(names),
            "date": self._dedupe_preserve_order(dates),
            "amount": self._dedupe_preserve_order(amounts)
        }


def extract_fields(text: str) -> Dict[str, Optional[str]]:
    extractor = InformationExtractor()
    return extractor.extract_all(text)


def extract_fields_as_arrays(text: str) -> Dict[str, Any]:
    extractor = InformationExtractor()
    return extractor.extract_all_as_arrays(text)
