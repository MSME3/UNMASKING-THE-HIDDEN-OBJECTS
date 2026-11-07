"""
Enhanced View-Invariant & Perspective-Rectified License Plate Engine.
Supports multi-angle car and motorcycle plate localization, deskewing, fuzzy matching,
and cross-viewpoint recognition.
"""
import re
import cv2
import numpy as np
from difflib import SequenceMatcher
from collections import Counter

# Common OCR confusion character map
_OCR_CONFUSIONS = {
    'O': '0', 'D': '0', 'Q': '0', 'U': '0',
    'I': '1', 'L': '1', 'J': '1', '|': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
}


def normalize_ocr_text(text: str) -> str:
    """
    Strips spaces, punctuation, and non-alphanumerics, and standardizes to uppercase.
    Example: "MH-12 AB 1234" -> "MH12AB1234"
    """
    if not text:
        return ""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()


def canonicalize_for_comparison(text: str) -> str:
    """
    Replaces visually ambiguous characters (e.g., 'O' vs '0', 'I' vs '1')
    to maximize fuzzy matching resilience across camera angles.
    """
    cleaned = normalize_ocr_text(text)
    return "".join(_OCR_CONFUSIONS.get(c, c) for c in cleaned)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral coordinates: [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies homography / perspective transformation to deskew an angled plate into a flat rectangle.
    """
    rect = order_quad_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_a), int(width_b), 120)

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b), 40)

    # Standardized normalized target plane
    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped


class TemporalPlateTracker:
    """
    Accumulates plate observations across consecutive live video frames
    to resolve temporary glare, motion blur, and turning angles.
    """

    def __init__(self, max_history=8):
        self.max_history = max_history
        self.history = []

    def add(self, plate_text: str):
        if plate_text and len(plate_text) >= 4:
            self.history.append(plate_text)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def consensus(self) -> str:
        if not self.history:
            return ""
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]


_EASYOCR_READER = None
_EASYOCR_INIT_ATTEMPTED = False
_OCR_DIAGNOSTICS_LOGGED = {
    "pytesseract": False,
    "easyocr": False,
    "general": False,
}


def _get_easyocr_reader():
    global _EASYOCR_READER, _EASYOCR_INIT_ATTEMPTED
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    if _EASYOCR_INIT_ATTEMPTED:
        return None
    _EASYOCR_INIT_ATTEMPTED = True
    try:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
        return _EASYOCR_READER
    except Exception as e:
        if not _OCR_DIAGNOSTICS_LOGGED["easyocr"]:
            _OCR_DIAGNOSTICS_LOGGED["easyocr"] = True
            print(f"[LPR-OCR] Diagnostic: easyocr is unavailable in current environment ({e}).")
        return None


class LicensePlateVerifier:
    """
    Multi-View & Angle-Tolerant License Plate Verifier for NVIDIA Jetson Nano.
    Performs multi-aspect plate localization, perspective deskewing,
    contrast enhancement, and fuzzy alphanumeric matching.
    """

    def __init__(self, similarity_threshold: float = 0.82):
        self.similarity_threshold = similarity_threshold
        self.temporal_trackers = {}

    def locate_and_rectify_plate(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        """
        Extracts and perspective-rectifies plate candidates from any vehicle view (front, rear, side).
        Supports:
          - Single-line car plates (aspect 2.5 - 5.5)
          - Square / 2-line bike plates (aspect 1.1 - 2.4)
          - Angled perspective views via 4-point homography transform
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]
        if h < 25 or w < 35:
            return [vehicle_crop]

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for dark/harsh lighting
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)

        candidates = []
        
        # Multi-scale morphological kernels (horizontal and block kernels for 1-line and 2-line plates)
        for k_w, k_h in [(17, 3), (11, 7)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
            top_hat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
            black_hat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
            grad = cv2.add(cv2.subtract(blurred, black_hat), top_hat)

            grad_x = cv2.Sobel(grad, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            grad_x = np.absolute(grad_x)
            min_v, max_v = np.min(grad_x), np.max(grad_x)
            if max_v > min_v:
                grad_x = 255 * ((grad_x - min_v) / (max_v - min_v))
            grad_x = grad_x.astype("uint8")

            _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (0.008 * w * h) or area > (0.45 * w * h):
                    continue

                min_rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(min_rect)
                box = np.int0(box)

                (cw, ch) = min_rect[1]
                if cw == 0 or ch == 0:
                    continue

                aspect = max(cw, ch) / min(cw, ch)
                # Broad aspect ratio range covering single-line cars (2.5-5.5) and 2-line bikes (1.1-2.4)
                if 1.1 <= aspect <= 5.8:
                    try:
                        warped = four_point_transform(vehicle_crop, box.astype("float32"))
                        if warped is not None and warped.size > 0:
                            candidates.append(warped)
                    except Exception:
                        pass

        # If contour detection found valid rectified patches, return top candidates
        if candidates:
            return candidates[:3]

        # Robust Heuristic Fallbacks for Front/Rear views
        fallbacks = [
            # Lower 60% (standard rear car/bike plate location)
            vehicle_crop[int(h * 0.40):int(h * 0.95), int(w * 0.10):int(w * 0.90)],
            # Front bumper / grill middle 50%
            vehicle_crop[int(h * 0.30):int(h * 0.80), int(w * 0.15):int(w * 0.85)]
        ]
        return [f for f in fallbacks if f.size > 0]

    def extract_plate_text(self, vehicle_crop: np.ndarray, try_mirror: bool = True) -> str:
        """
        Extracts plate alphanumeric string across multi-angle candidate crops.
        Automatically tries both the original and horizontally flipped (mirrored) version,
        which fixes the reversed plate text observed in live CSI camera feeds.
        """
        candidates = self.locate_and_rectify_plate(vehicle_crop)
        if not candidates:
            return ""

        extracted_strings = []

        for crop in candidates:
            h, w = crop.shape[:2]
            target_h = 64
            target_w = max(140, int(w * (target_h / max(h, 1))))
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Build list of image variants to try: original + horizontally flipped
            # Horizontal flip (flipCode=1) corrects mirrored live camera feeds
            gray_variants = [gray]
            if try_mirror:
                gray_variants.append(cv2.flip(gray, 1))

            for gray_img in gray_variants:
                text = ""

                # Strategy 1: Tesseract OCR (Single-line and 2-line bike formats)
                try:
                    import pytesseract
                    for psm in (7, 8, 6):  # 7=single line, 8=single word, 6=2-line block
                        config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        t = pytesseract.image_to_string(gray_img, config=config)
                        cleaned = normalize_ocr_text(t)
                        if len(cleaned) >= 4:
                            text = cleaned
                            break
                except Exception as e:
                    # Fallback: direct subprocess execution of system tesseract binary
                    try:
                        import shutil
                        import subprocess
                        tess_bin = shutil.which("tesseract")
                        if tess_bin:
                            ok_enc, buf = cv2.imencode(".png", gray_img)
                            if ok_enc:
                                for psm in (7, 8, 6):
                                    cmd = [
                                        tess_bin, "stdin", "stdout",
                                        "--psm", str(psm),
                                        "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                                    ]
                                    proc = subprocess.run(
                                        cmd, input=buf.tobytes(),
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        timeout=4
                                    )
                                    if proc.returncode == 0:
                                        cleaned = normalize_ocr_text(proc.stdout.decode("utf-8", errors="ignore"))
                                        if len(cleaned) >= 4:
                                            text = cleaned
                                            break
                    except Exception as sub_e:
                        if not _OCR_DIAGNOSTICS_LOGGED["pytesseract"]:
                            _OCR_DIAGNOSTICS_LOGGED["pytesseract"] = True
                            print(f"[LPR-OCR] Diagnostic: tesseract failed ({sub_e}).")

                # Strategy 2: EasyOCR fallback (cached singleton)
                if not text:
                    try:
                        reader = _get_easyocr_reader()
                        if reader is not None:
                            res = reader.readtext(gray_img, detail=0)
                            text = normalize_ocr_text("".join(res))
                    except Exception as e:
                        if not _OCR_DIAGNOSTICS_LOGGED["easyocr"]:
                            _OCR_DIAGNOSTICS_LOGGED["easyocr"] = True
                            print(f"[LPR-OCR] Diagnostic: easyocr execution failed ({e}).")

                if text and len(text) >= 4:
                    extracted_strings.append(text)

        if not extracted_strings and _OCR_DIAGNOSTICS_LOGGED["pytesseract"] and _OCR_DIAGNOSTICS_LOGGED["easyocr"]:
            if not _OCR_DIAGNOSTICS_LOGGED["general"]:
                _OCR_DIAGNOSTICS_LOGGED["general"] = True
                print("[LPR-OCR] Notice: No active OCR engine is installed (pytesseract/easyocr). License plate text extraction returned empty string.")

        if extracted_strings:
            return max(extracted_strings, key=len)
        return ""

    def compare_plates(self, query_plate: str, candidate_plate: str) -> tuple[bool, float]:
        """
        Compares query and candidate license plate strings with OCR confusion tolerance,
        substring matching, and suffix matching.
        Returns: (is_match: bool, similarity_ratio: float)
        """
        norm_q = canonicalize_for_comparison(query_plate)
        norm_c = canonicalize_for_comparison(candidate_plate)

        if not norm_q or not norm_c:
            return False, 0.0

        if norm_q == norm_c:
            return True, 1.0

        # Substring match (handles cases where front camera captured "12AB1234" vs rear "MH12AB1234")
        if len(norm_c) >= 5 and (norm_c in norm_q or norm_q in norm_c):
            return True, 0.92

        # Suffix matching (vehicle number digits e.g. "AB1234" match)
        if len(norm_q) >= 4 and len(norm_c) >= 4:
            if norm_q[-4:] == norm_c[-4:]:
                return True, 0.88

        ratio = SequenceMatcher(None, norm_q, norm_c).ratio()
        return ratio >= self.similarity_threshold, ratio

    def verify_candidate(self, candidate_crop: np.ndarray, query_plate: str, target_type: str) -> dict:
        """
        Two-stage post-processing verifier:
        Called on vehicle candidates to verify license plate consistency.
        
        Returns:
            dict with:
              - plate_status: "verified" | "mismatch" | "unreadable" | "not_applicable"
              - candidate_plate: str
              - similarity_ratio: float
        """
        if target_type == "person" or not query_plate:
            return {
                "plate_status": "not_applicable",
                "candidate_plate": "",
                "similarity_ratio": 0.0,
            }

        candidate_plate = self.extract_plate_text(candidate_crop)
        if not candidate_plate:
            return {
                "plate_status": "unreadable",
                "candidate_plate": "",
                "similarity_ratio": 0.0,
            }

        is_match, ratio = self.compare_plates(query_plate, candidate_plate)
        return {
            "plate_status": "verified" if is_match else "mismatch",
            "candidate_plate": candidate_plate,
            "similarity_ratio": ratio,
        }
