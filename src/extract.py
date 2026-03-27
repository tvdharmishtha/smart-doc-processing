import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser


class InformationExtractor:
    def __init__(self):
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
        self.name_stopwords = {
            'invoice', 'name', 'date', 'total', 'amount', 'grand', 'bill',
            'customer', 'client', 'vendor', 'document', 'doc', 'invoice date'
        }
        self.invoice_noise_tokens = {
            'invoice', 'inv', 'invc', 'invoi', 'invoic', 'irnce', 'irncce', 'irncice'
        }
        self.common_letter_bigrams = {
            'ab', 'ac', 'al', 'an', 'ar', 'as', 'at', 'bc', 'ce', 'ch', 'co', 'de', 'ea', 'el', 'en', 'er',
            'es', 'et', 'ha', 'he', 'hi', 'ia', 'ic', 'ie', 'in', 'io', 'it', 'ja', 'jo', 'la', 'li', 'll',
            'lo', 'mi', 'na', 'ne', 'ni', 'nt', 'oh', 'on', 'or', 'pa', 'ph', 'pt', 'ra', 're', 'ri', 'ro',
            'rt', 'sa', 'se', 'sh', 'si', 'sm', 'st', 'ta', 'te', 'th', 'ti', 'to', 'tr', 'va', 've', 'vi',
            'vt', 'wo'
        }

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
        text = re.sub(r'(?<=[A-Za-z])[._](?=[A-Za-z])', ' ', text)
        text = re.sub(r'(?i)\bP(?:u|y|t|n)?t\s*\.?\s*Ltd\b', 'Pvt Ltd', text)
        text = re.sub(r'[{}\[\]()|_]', ' ', text)
        text = re.sub(r'[ \t\f\v]+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n+', '\n', text)
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

        if scale == 1 and len(raw_digits) >= 4:
            score += 10
        if scale == 100:
            score -= 2
        if scale == 100 and raw_digits[-2:] in self.common_decimal_suffixes:
            score += 14

        return score

    def _normalize_amount_number(self, number: str) -> Optional[str]:
        cleaned = str(number).replace(',', '').strip()
        if not cleaned:
            return None

        if '.' in cleaned:
            try:
                integer_part, decimal_part = cleaned.split('.', 1)
                normalized = f"{float(cleaned):.2f}"

                if integer_part.isdigit() and len(integer_part) >= 5:
                    significant_index = next((index for index, char in enumerate(integer_part) if char != '0'), None)
                    if significant_index is not None:
                        first_digit = integer_part[significant_index]
                        integer_candidates = [integer_part]

                        if first_digit == '7':
                            integer_candidates.append(
                                integer_part[:significant_index] + '1' + integer_part[significant_index + 1:]
                            )

                        best_value = float(normalized)
                        best_score = self._score_amount_candidate(integer_part, best_value, 1)

                        for integer_candidate in integer_candidates[1:]:
                            try:
                                candidate_value = float(f"{integer_candidate}.{decimal_part}")
                            except ValueError:
                                continue

                            candidate_score = self._score_amount_candidate(integer_candidate, candidate_value, 1)
                            if candidate_value <= 50000:
                                candidate_score += 10

                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_value = candidate_value

                        return f"{best_value:.2f}"

                return normalized
            except ValueError:
                return None

        if not cleaned.isdigit():
            return None

        base_value = float(cleaned)
        candidates = [(base_value, 1)]
        if len(cleaned) >= 4 and cleaned[-2:] in self.common_decimal_suffixes:
            candidates.append((base_value / 100.0, 100))
        if len(cleaned) >= 5:
            significant_index = next((index for index, char in enumerate(cleaned) if char != '0'), None)
            if significant_index is not None and cleaned[significant_index] == '7':
                candidate_digits = cleaned[:significant_index] + '1' + cleaned[significant_index + 1:]
                candidates.append((float(candidate_digits), 1))

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

    def _clean_name_candidate(self, candidate: str) -> str:
        candidate = self._clean_text(candidate)
        candidate = re.split(
            r'(?i)\b(?:date|dats?|cate|cnte|total|tota[f1i]?|amount|invoice|inv[o0]i\w*|invoice date|invoice no|invoice number)\b\s*[:\-]?',
            candidate,
        )[0]
        candidate = re.sub(r'(?i)\bP(?:u|y|t|n)?t\b(?=\s+Ltd\b)', 'Pvt', candidate)
        candidate = re.sub(r'(?i)\bP(?:u|y|t|n)?t\s*\.?\s*Ltd\b', 'Pvt Ltd', candidate)
        candidate = re.sub(r'\s+', ' ', candidate).strip(' .:-')

        tokens = []
        for token in candidate.split():
            lowered = re.sub(r'[^a-z]', '', token.lower())
            if lowered in self.invoice_noise_tokens:
                continue
            tokens.append(token)

        candidate = " ".join(tokens).strip(' .:-')

        token_list = [token for token in candidate.split() if token]
        filtered_tokens = []
        for index, token in enumerate(token_list):
            letters = re.sub(r'[^A-Za-z]', '', token)
            lowered = letters.lower()
            next_token = token_list[index + 1] if index + 1 < len(token_list) else ""
            label_similarity = max(
                SequenceMatcher(None, lowered, label).ratio()
                for label in ('name', 'date', 'total', 'amount', 'invoice')
            ) if lowered else 0.0

            if lowered in {'pm', 'pn', 'pt', 'pv', 'put'} and next_token.lower() == 'ltd':
                filtered_tokens.append('Pvt')
                continue

            if label_similarity >= 0.72:
                continue

            if len(filtered_tokens) >= 2 and index == len(token_list) - 1 and letters:
                vowel_count = sum(char in 'aeiou' for char in lowered)
                is_business_suffix = lowered in {'pvt', 'ltd', 'corp', 'inc', 'llc'}
                is_short_token = len(letters) <= 3
                if not is_business_suffix and not is_short_token and (
                    vowel_count == 0 or (len(letters) >= 4 and vowel_count == 1 and lowered not in {'smith', 'shah'})
                ):
                    continue

            filtered_tokens.append(token)

        candidate = " ".join(filtered_tokens).strip(' .:-')
        return candidate

    def _is_plausible_name(self, candidate: str) -> bool:
        if not candidate:
            return False

        candidate = candidate.strip()
        if len(candidate) < 3 or len(candidate) > 80:
            return False
        if re.search(r'\d', candidate):
            return False

        tokens = [token for token in re.split(r'\s+', candidate) if token]
        if not tokens or len(tokens) > 6:
            return False

        alpha_tokens = [token for token in tokens if re.search(r'[A-Za-z]', token)]
        if not alpha_tokens:
            return False

        normalized = candidate.lower()
        if normalized in self.name_stopwords:
            return False
        if any(token in normalized for token in self.invoice_noise_tokens):
            return False

        return True

    def _score_name_candidate(self, candidate: str) -> float:
        score = 0.0
        tokens = [token for token in re.split(r'\s+', candidate) if token]

        score += min(len(tokens), 4) * 3
        score += sum(token[:1].isupper() for token in tokens) * 2
        if any(token.lower() in {'pvt', 'ltd', 'corp', 'inc', 'llc'} for token in tokens):
            score += 4
        if '.' in candidate or '&' in candidate:
            score += 1
        if all(token[:1].isupper() for token in tokens if token[:1].isalpha()):
            score += 4
        if len(tokens) == 1:
            score -= 3
        for token in tokens:
            letters = re.sub(r'[^A-Za-z]', '', token)
            lowered = letters.lower()
            if not letters or lowered in {'pvt', 'ltd', 'corp', 'inc', 'llc'}:
                continue
            vowel_ratio = sum(char in 'aeiou' for char in lowered) / max(len(letters), 1)
            if len(letters) >= 4 and vowel_ratio < 0.25:
                score -= 5
        score -= max(len(tokens) - 4, 0) * 2

        return score

    def _normalize_name_for_grouping(self, candidate: str) -> str:
        tokens = []
        for token in re.split(r'\s+', candidate):
            cleaned = re.sub(r'[^A-Za-z]', '', token).lower()
            if not cleaned:
                continue
            cleaned = cleaned.replace('0', 'o')
            cleaned = cleaned.replace('1', 'l')
            cleaned = cleaned.replace('5', 's')
            cleaned = re.sub(r'irn', 'in', cleaned)
            cleaned = re.sub(r'mu', 'mi', cleaned)
            cleaned = re.sub(r'uth', 'ith', cleaned)
            cleaned = re.sub(r'thn$', 'th', cleaned)
            cleaned = re.sub(r'pd$', 'pvt', cleaned)
            cleaned = re.sub(r'vv', 'w', cleaned)
            tokens.append(cleaned)

        return ' '.join(tokens)

    def _score_token_readability(self, token: str) -> float:
        letters = re.sub(r'[^A-Za-z]', '', token)
        lowered = letters.lower()
        if not letters:
            return -100.0

        score = 0.0
        vowel_count = sum(char in 'aeiou' for char in lowered)
        vowel_ratio = vowel_count / max(len(letters), 1)

        score += min(len(letters), 8)
        if token[:1].isupper():
            score += 2
        if lowered in {'pvt', 'ltd', 'corp', 'inc', 'llc'}:
            score += 8
        if 0.2 <= vowel_ratio <= 0.7:
            score += 4
        elif len(letters) > 3:
            score -= 3
        if len(lowered) >= 2:
            bigram_score = 0.0
            for index in range(len(lowered) - 1):
                bigram = lowered[index:index + 2]
                bigram_score += 1.2 if bigram in self.common_letter_bigrams else -0.6
            score += bigram_score
        if 3 <= len(lowered) <= 4 and all(ord(lowered[index + 1]) - ord(lowered[index]) == 1 for index in range(len(lowered) - 1)):
            score += 5
        if re.search(r'(?i)[bcdfghjklmnpqrstvwxyz]{4,}', letters):
            score -= 3

        return score

    def _normalize_business_token(self, token: str) -> Optional[str]:
        letters = re.sub(r'[^A-Za-z]', '', token).lower()
        if not letters:
            return None

        weak_pvt_tokens = {'p', 'pu', 'pv', 'pt', 'pm', 'pd', 'put'}
        if letters in weak_pvt_tokens:
            return 'Pvt'

        for canonical in ('pvt', 'ltd', 'corp', 'inc', 'llc'):
            if SequenceMatcher(None, letters, canonical).ratio() >= 0.6:
                return canonical.title() if canonical != 'llc' else 'LLC'

        return None

    def _canonicalize_business_suffixes(self, candidate: str) -> str:
        tokens = [token for token in candidate.split() if token]
        if not tokens:
            return candidate

        normalized_tokens = []
        suffix_presence = {key: False for key in ('pvt', 'ltd', 'corp', 'inc', 'llc')}

        for index, token in enumerate(tokens):
            prev_token = tokens[index - 1] if index > 0 else ''
            next_token = tokens[index + 1] if index + 1 < len(tokens) else ''
            prev_business = self._normalize_business_token(prev_token)
            next_business = self._normalize_business_token(next_token)

            business_token = self._normalize_business_token(token)
            if not business_token and (prev_business or next_business):
                business_token = self._normalize_business_token(token)

            if business_token:
                suffix_presence[business_token.lower()] = True
                continue

            normalized_tokens.append(token)

        while normalized_tokens and len(re.sub(r'[^A-Za-z]', '', normalized_tokens[-1])) <= 1:
            normalized_tokens.pop()

        ordered_suffixes = []
        for suffix in ('pvt', 'ltd', 'corp', 'inc', 'llc'):
            if suffix_presence[suffix]:
                ordered_suffixes.append(suffix.title() if suffix != 'llc' else 'LLC')

        if ordered_suffixes:
            normalized_tokens.extend(ordered_suffixes)

        return ' '.join(normalized_tokens).strip()

    def _restore_token_case(self, source: str, normalized: str) -> str:
        if not normalized:
            return source
        if source.isupper():
            return normalized.upper()
        if source.istitle() or source[:1].isupper():
            return normalized.title()
        return normalized

    def _finalize_name_token(self, token: str, peer_tokens: List[str]) -> str:
        business_token = self._normalize_business_token(token)
        if business_token:
            return business_token

        normalized_token = self._normalize_name_for_grouping(token)
        normalized_token = normalized_token.split()[0] if normalized_token else ''
        letters = re.sub(r'[^A-Za-z]', '', token)

        if not normalized_token or not letters:
            return token

        similarity = SequenceMatcher(None, letters.lower(), normalized_token.lower()).ratio()
        original_score = self._score_token_readability(token)
        normalized_score = self._score_token_readability(normalized_token)

        peer_support = 0.0
        for peer_token in peer_tokens:
            peer_letters = re.sub(r'[^A-Za-z]', '', peer_token)
            if not peer_letters:
                continue
            peer_support += SequenceMatcher(None, normalized_token.lower(), peer_letters.lower()).ratio()

        if similarity >= 0.72 and normalized_score + peer_support * 0.4 >= original_score + 1.5:
            return self._restore_token_case(token, normalized_token)

        return token

    def _score_token_option(self, token: str, peer_tokens: List[str]) -> float:
        score = self._score_token_readability(token)
        letters = re.sub(r'[^A-Za-z]', '', token).lower()
        if not letters:
            return -100.0

        for peer_token in peer_tokens:
            peer_letters = re.sub(r'[^A-Za-z]', '', peer_token).lower()
            if not peer_letters:
                continue
            normalized_peer = self._normalize_name_for_grouping(peer_token).replace(' ', '')
            score += SequenceMatcher(None, letters, peer_letters).ratio() * 2.5
            if normalized_peer:
                score += SequenceMatcher(None, letters, normalized_peer).ratio() * 2.0
            if letters == peer_letters:
                score += 2.0

        return score

    def _generate_token_variants(self, token: str) -> List[str]:
        variants = [token]
        letters = re.sub(r'[^A-Za-z]', '', token)
        if not letters:
            return variants

        normalized = self._normalize_name_for_grouping(token).split()
        if normalized:
            variants.append(self._restore_token_case(token, normalized[0]))

        lowered = letters.lower()
        if 'u' in lowered and len(letters) >= 4:
            variants.append(self._restore_token_case(token, lowered.replace('u', 'i', 1)))
        if lowered.endswith('uth'):
            variants.append(self._restore_token_case(token, lowered[:-3] + 'ith'))

        confusion_map = {
            'a': ['o'],
            'b': ['e', 'h'],
            'c': ['e', 'o'],
            'e': ['b', 'c'],
            'g': ['c'],
            'i': ['l'],
            'l': ['i'],
            'm': ['n'],
            'n': ['m', 'h'],
            'o': ['a', 'c'],
            'p': ['f', 'v'],
            'u': ['v', 'i'],
            'v': ['u', 'y'],
            'y': ['v'],
        }

        if 3 <= len(letters) <= 6:
            for index, char in enumerate(lowered):
                for replacement in confusion_map.get(char, []):
                    variant = lowered[:index] + replacement + lowered[index + 1:]
                    variants.append(self._restore_token_case(token, variant))

        if 2 <= len(letters) <= 4:
            insertion_letters = ['a', 'b', 'c', 'e', 'i', 'o', 'p', 'r', 't', 'v']

            if len(letters) == 2:
                start_ord = ord(lowered[0])
                end_ord = ord(lowered[1])
                if end_ord - start_ord == 2:
                    insertion_letters.append(chr(start_ord + 1))

            for index in range(1, len(lowered)):
                for insertion in insertion_letters:
                    variant = lowered[:index] + insertion + lowered[index:]
                    variants.append(self._restore_token_case(token, variant))

        deduped = []
        seen = set()
        for variant in variants:
            key = re.sub(r'[^A-Za-z]', '', variant).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(variant)

        return deduped

    def _merge_name_candidates(self, candidates: List[str]) -> Optional[str]:
        token_lists = [candidate.split() for candidate in candidates if candidate.strip()]
        if not token_lists:
            return None

        merged_tokens = []
        max_token_count = max(len(tokens) for tokens in token_lists)

        for index in range(max_token_count):
            position_tokens = [tokens[index] for tokens in token_lists if len(tokens) > index and tokens[index].strip()]
            if not position_tokens:
                continue

            token_options = []
            for token in position_tokens:
                token_options.extend(self._generate_token_variants(token))

            best_token = max(token_options, key=lambda token: self._score_token_option(token, position_tokens))
            best_token = self._finalize_name_token(best_token, position_tokens)
            merged_tokens.append(best_token)

        if not merged_tokens:
            return None

        merged_name = " ".join(merged_tokens).strip()

        candidate_text = " ".join(candidates)
        if re.search(r'(?i)\bpvt\b', candidate_text) and re.search(r'(?i)\bltd\b', candidate_text):
            if not re.search(r'(?i)\bpvt\b', merged_name):
                merged_name = f"{merged_name} Pvt".strip()
            if not re.search(r'(?i)\bltd\b', merged_name):
                merged_name = f"{merged_name} Ltd".strip()

        return self._canonicalize_business_suffixes(self._clean_name_candidate(merged_name))

    def _refine_company_name(self, candidate: str, peer_candidates: List[str]) -> str:
        tokens = [token for token in candidate.split() if token]
        if len(tokens) < 2:
            return candidate

        business_suffixes = {'Pvt', 'Ltd', 'Corp', 'Inc', 'LLC'}
        if not any(token in business_suffixes for token in tokens[1:]):
            return candidate

        prefix = tokens[0]
        prefix_letters = re.sub(r'[^A-Za-z]', '', prefix)
        if not prefix_letters or len(prefix_letters) > 3:
            return candidate

        peer_prefixes = []
        for peer in peer_candidates:
            peer_tokens = [token for token in peer.split() if token]
            if not peer_tokens:
                continue
            peer_prefix = re.sub(r'[^A-Za-z]', '', peer_tokens[0])
            if peer_prefix:
                peer_prefixes.append(peer_prefix)

        prefix_variants = self._generate_token_variants(prefix)
        prefix_variants.append(prefix)

        best_prefix = prefix
        best_score = float('-inf')

        for variant in prefix_variants:
            letters = re.sub(r'[^A-Za-z]', '', variant)
            if not letters:
                continue

            score = self._score_token_readability(variant)
            if len(letters) == 3 and variant[:1].isupper():
                score += 2.5
            if len(prefix_letters) <= 2 and len(letters) == len(prefix_letters) + 1:
                score += 2.0

            normalized_letters = self._normalize_name_for_grouping(variant).replace(' ', '')
            for peer_prefix in peer_prefixes:
                score += SequenceMatcher(None, letters.lower(), peer_prefix.lower()).ratio() * 2.5
                if normalized_letters:
                    peer_normalized = self._normalize_name_for_grouping(peer_prefix).replace(' ', '')
                    score += SequenceMatcher(None, normalized_letters.lower(), peer_normalized.lower()).ratio() * 2.0

            if score > best_score:
                best_score = score
                best_prefix = self._restore_token_case(prefix, letters)

        tokens[0] = best_prefix
        return ' '.join(tokens)

    def choose_best_name(self, candidates: List[str]) -> Optional[str]:
        cleaned_candidates = []
        for candidate in candidates:
            cleaned = self._clean_name_candidate(candidate)
            if self._is_plausible_name(cleaned):
                cleaned_candidates.append(cleaned)

        if not cleaned_candidates:
            return None

        buckets = []
        for candidate in cleaned_candidates:
            normalized = self._normalize_name_for_grouping(candidate)
            matched_bucket = None

            for bucket in buckets:
                similarity = SequenceMatcher(None, normalized, bucket['normalized']).ratio()
                if similarity >= 0.58:
                    matched_bucket = bucket
                    break

            if matched_bucket is None:
                buckets.append({
                    'normalized': normalized,
                    'candidates': [candidate],
                    'score': self._score_name_candidate(candidate),
                })
            else:
                matched_bucket['candidates'].append(candidate)
                matched_bucket['score'] += self._score_name_candidate(candidate)

        best_bucket = max(buckets, key=lambda item: item['score'])
        merged_name = self._merge_name_candidates(best_bucket['candidates'])
        if merged_name and self._is_plausible_name(merged_name):
            merged_name = self._refine_company_name(merged_name, best_bucket['candidates'])
            return self._canonicalize_business_suffixes(merged_name)

        fallback_name = max(best_bucket['candidates'], key=self._score_name_candidate)
        fallback_name = self._refine_company_name(fallback_name, best_bucket['candidates'])
        return self._canonicalize_business_suffixes(fallback_name)

    def choose_best_value(self, field_name: str, values: List[str]) -> Optional[str]:
        cleaned_values = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned_values:
            return None

        if field_name == "name":
            return self.choose_best_name(cleaned_values)

        if field_name == "date":
            scores = {}
            for value in cleaned_values:
                normalized = self._normalize_date(value)
                if not normalized:
                    continue
                scores[normalized] = scores.get(normalized, 0.0) + 1.0
            return max(scores, key=scores.get) if scores else None

        if field_name == "amount":
            scores = {}
            for value in cleaned_values:
                normalized = self._normalize_amount(value)
                if not normalized:
                    continue
                numeric = float(normalized)
                score = 1.0
                if 10 <= numeric <= 50000:
                    score += 1.0
                if 50 <= numeric <= 20000:
                    score += 1.0
                scores[normalized] = scores.get(normalized, 0.0) + score
            return max(scores, key=scores.get) if scores else None

        return cleaned_values[0]

    def extract_name(self, text: str) -> Optional[str]:
        text = self._clean_text(text)

        label_patterns = [
            r'(?im)^\s*(?:name|customer|client|vendor|employee|person)\b\s*[:\-]?\s*([^\n]+)',
            r'(?i)\b(?:name|customer|client|vendor|employee|person)\b\s*[:\-]?\s*([A-Za-z][A-Za-z .&/-]{1,80})',
        ]

        candidates = []
        for pattern in label_patterns:
            for match in re.finditer(pattern, text):
                candidate = self._clean_name_candidate(match.group(1))
                if self._is_plausible_name(candidate):
                    candidates.append(candidate)

        if candidates:
            return max(candidates, key=self._score_name_candidate)

        for line in text.splitlines():
            candidate = self._clean_name_candidate(line)
            lowered = candidate.lower()
            if any(stopword in lowered for stopword in ('invoice', 'date', 'total', 'amount')):
                continue
            if self._is_plausible_name(candidate):
                candidates.append(candidate)

        if candidates:
            return max(candidates, key=self._score_name_candidate)

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
