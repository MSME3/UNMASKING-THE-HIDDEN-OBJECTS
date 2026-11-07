"""
AI Engine: InsightFace (Person), YOLOv8 (Detector), and DINOv2 + HSV (Vehicle Re-ID).
"""
import gc
import threading
import time
from collections import Counter
import cv2
import numpy as np
from PIL import Image

from .config import (
    FACE_THRESHOLD,
    VEHICLE_THRESHOLD,
    VEHICLE_COLOR_THRESHOLD,
    VEHICLE_DINO_MIN,
    VEHICLE_PLATE_PROMOTION_THRESHOLD,
    VEHICLE_CONF_MIN,
    VEHICLE_YOLO_CLASSES,
    YOLO_MODEL_PATH,
    ENABLE_LPR,
    LPR_SIMILARITY_THRESHOLD,
    INSIGHTFACE_ROOT,
    TORCH_HOME,
    TWO_WHEEL_TYPES,
    is_compatible_target_type,
)
from .license_plate import LicensePlateVerifier


class Engine:
    """
    Unified AI inference engine:
      - InsightFace (buffalo_l) for deep facial identity embeddings (512-dim).
      - YOLOv8n for fast person/car/motorcycle/bus/truck localization.
      - DINOv2 (vits14, 384-dim) + 128-bin HSV color histogram hybrid Re-ID for vehicles (512-dim).
    """

    def __init__(self, use_gpu=True, skip_vehicle_reid=False,
                 face_threshold=FACE_THRESHOLD, vehicle_threshold=VEHICLE_THRESHOLD,
                 enable_lpr=None):
        import torch
        import torchvision.transforms as T
        from insightface.app import FaceAnalysis
        from ultralytics import YOLO

        self._infer_lock = threading.Lock()
        self._infer_count = 0

        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        print(f"[Engine] device={self.device}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        print("[Engine] loading InsightFace (person identity)...")
        self.face_app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_ROOT, providers=providers)
        self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

        print(f"[Engine] loading YOLOv8n detector from {YOLO_MODEL_PATH}...")
        self.yolo = YOLO(YOLO_MODEL_PATH)

        self.torch = torch
        self.skip_vehicle_reid = skip_vehicle_reid
        self.face_threshold = face_threshold
        self.vehicle_threshold = vehicle_threshold
        self.vehicle_dino_min = VEHICLE_DINO_MIN
        self.enable_lpr = ENABLE_LPR if enable_lpr is None else enable_lpr
        self.plate_verifier = LicensePlateVerifier(similarity_threshold=LPR_SIMILARITY_THRESHOLD)
        print(f"[Engine] match thresholds -> person: {self.face_threshold:.2f} "
              f"| vehicle(car/bike): {self.vehicle_threshold:.2f} (dino_min: {self.vehicle_dino_min:.2f}, LPR enabled: {self.enable_lpr})")

        self.dinov2 = None
        if skip_vehicle_reid:
            print("[Engine] --skip-vehicle-reid set: car/bike boxes will be detected but "
                  "NOT matched (person search works fully).")
        else:
            self.dinov2 = self._load_dinov2_with_retry()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[Engine] ready.")

    def _load_dinov2_with_retry(self, attempts=3, wait_sec=5):
        for attempt in range(1, attempts + 1):
            try:
                print(f"[Engine] loading DINOv2 (vehicle Re-ID features)... attempt {attempt}/{attempts}")
                return self.torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()
            except Exception as e:
                print(f"[Engine] DINOv2 load failed: {e}")
                if attempt < attempts:
                    time.sleep(wait_sec)
        print(
            "[Engine] Could not load DINOv2 after retries - continuing in PERSON-ONLY mode.\n"
            "  Car/bike boxes will still be detected by YOLO but won't be matched.\n"
        )
        self.skip_vehicle_reid = True
        return None

    def extract_vehicle_color_signature(self, crop_img):
        """
        Extracts a normalized 128-bin HSV color histogram (16 Hue x 4 Sat x 2 Val)
        focused specifically on the vehicle body paint region (excluding ground shadow and sky).
        """
        if crop_img is None or crop_img.size == 0:
            return np.zeros(128, dtype=np.float32)
        resized = cv2.resize(crop_img, (160, 160))
        h, w = resized.shape[:2]
        body = resized[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
        if body.size == 0:
            body = resized

        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        # 16 Hue bins (0-180), 4 Saturation bins (0-256), 2 Value bins (0-256) -> 128 total bins
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 2], [0, 180, 0, 256, 0, 256]).flatten()
        return hist / (np.linalg.norm(hist) + 1e-6)

    def extract_vehicle_hybrid_embedding(self, crop_img):
        """
        Unified 512-dim Vehicle Hybrid Re-ID embedding:
          - 384-dim full DINOv2 deep structural ViT features (70% cosine similarity weight).
          - 128-dim HSV perceptual color histogram (30% cosine similarity weight).
          - Mathematically guaranteed: dot(v1, v2) = 0.70 * dot(dino1, dino2) + 0.30 * dot(color1, color2).
          - Total vector length = 384 + 128 = 512, keeping FAISS index compatibility.
        """
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            dino_raw = self.dinov2(tensor_img).cpu().numpy().flatten()
        # 384-dim normalized DINOv2 feature
        dino_emb = dino_raw / (np.linalg.norm(dino_raw) + 1e-6)

        # 128-dim normalized HSV body color histogram
        color_emb = self.extract_vehicle_color_signature(crop_img)

        # Exact 70% DINOv2 + 30% HSV cosine weighting
        fused = np.hstack((np.sqrt(0.70) * dino_emb, np.sqrt(0.30) * color_emb))
        return fused / (np.linalg.norm(fused) + 1e-6)

    def compute_vehicle_color_similarity(self, sig1_or_crop1, sig2_or_crop2):
        """
        Computes 3D HSV color similarity between two vehicle signatures or image crops.
        Returns a cosine similarity value between 0.0 and 1.0.
        """
        if isinstance(sig1_or_crop1, np.ndarray) and len(sig1_or_crop1.shape) == 3:
            s1 = self.extract_vehicle_color_signature(sig1_or_crop1)
        else:
            s1 = sig1_or_crop1

        if isinstance(sig2_or_crop2, np.ndarray) and len(sig2_or_crop2.shape) == 3:
            s2 = self.extract_vehicle_color_signature(sig2_or_crop2)
        else:
            s2 = sig2_or_crop2

        if s1 is None or s2 is None:
            return 1.0
        return float(np.dot(s1, s2))

    def verify_vehicle_target_match(self, query_target, cand_crop, score, context_info="", query_id=""):
        """
        Evaluates whether a candidate vehicle crop matches a target query representation.
        Applies calibrated visual threshold, 3D HSV color threshold, and license plate verification.
        Applies crop quality gating, decoupled structural DINOv2 cosine similarity,
        3D HSV color threshold, and strict license plate verification.
        Outputs structured diagnostic logging.
        """
        plate_text = ""
        plate_status = "not_applicable"
        # 1. Quality & Resolution Gate on Candidate Crop
        if cand_crop is not None and cand_crop.size > 0:
            ch, cw = cand_crop.shape[:2]
            if ch < 45 or cw < 45 or (ch * cw) < 2000:
                decision_reason = f"Candidate crop quality/size insufficient ({cw}x{ch} < 45x45)"
                qid_str = f"query={query_id[:8]} " if query_id else ""
                ctx_str = f"[{context_info}] " if context_info else ""
                print(f"[target-filter] REJECTED {ctx_str}{qid_str}reason='{decision_reason}'")
                return False, "", "unreadable", 0.0, decision_reason

        # 2. Fresh Embedding & Metric Decomposition
        eval_score = float(score)
        dino_sim = None
        color_sim = 1.0

        if cand_crop is not None and cand_crop.size > 0 and query_target.get("color_sig") is not None:
            color_sim = self.compute_vehicle_color_similarity(query_target["color_sig"], cand_crop)
        if cand_crop is not None and cand_crop.size > 0:
            if query_target.get("emb") is not None and getattr(self, "dinov2", None) is not None:
                cand_emb = self.extract_vehicle_hybrid_embedding(cand_crop)
                eval_score = float(np.dot(query_target["emb"], cand_emb))
                q_dino = query_target["emb"][:384] / np.sqrt(0.70)
                c_dino = cand_emb[:384] / np.sqrt(0.70)
                dino_sim = float(np.dot(q_dino, c_dino))
                color_sim = float(np.dot(query_target["emb"][384:], cand_emb[384:])) / 0.30
            elif query_target.get("color_sig") is not None:
                color_sim = self.compute_vehicle_color_similarity(query_target["color_sig"], cand_crop)

        # 3. License Plate Verification
        plate_text = ""
        plate_status = "not_applicable"
        query_plate = query_target.get("plate_text", "")
        if getattr(self, "enable_lpr", False) and cand_crop is not None and cand_crop.size > 0:
            lpr_res = self.plate_verifier.verify_candidate(
                cand_crop, query_plate, query_target.get("type", "car")
            )
            plate_text = lpr_res.get("candidate_plate", "")
            plate_status = lpr_res.get("plate_status", "not_applicable")

        is_match = False
        decision_reason = ""
        target_thresh = query_target.get("thresh", self.vehicle_threshold)

        # 4. Decoupled Decision Rules
        # Rule 1: License Plate Verified (allows cross-view matching promotion)
        if plate_status == "verified" and eval_score >= VEHICLE_PLATE_PROMOTION_THRESHOLD:
            is_match = True
            decision_reason = f"Plate verified ('{plate_text}' == '{query_plate}')"

        # Rule 2: License Plate Mismatch (strict rejection unless extreme visual similarity >= 0.95)
        elif plate_status == "mismatch":
            if eval_score >= 0.95:
                is_match = True
                decision_reason = "High visual similarity despite plate mismatch"
            else:
                is_match = False
                decision_reason = f"Plate mismatch ('{plate_text}' != '{query_plate}')"

        # Rule 3: Exact Same Vehicle Visual & Color Match (Plate unreadable / missing / not applicable)
        else:
            reasons = []
            if eval_score < target_thresh:
                reasons.append(f"Visual score {eval_score:.3f} < threshold {target_thresh:.2f}")
            if color_sim < VEHICLE_COLOR_THRESHOLD:
                reasons.append(f"Color score {color_sim:.3f} < threshold {VEHICLE_COLOR_THRESHOLD:.2f}")
            dino_min_val = getattr(self, "vehicle_dino_min", VEHICLE_DINO_MIN)
            if dino_sim is not None and dino_sim < dino_min_val:
                reasons.append(f"Structural DINOv2 {dino_sim:.3f} < threshold {dino_min_val:.2f}")

            if not reasons:
                is_match = True
                decision_reason = f"Visual & color match satisfied (plate: {plate_status})"
            else:
                is_match = False
                decision_reason = ", ".join(reasons)

        status_tag = "ACCEPTED" if is_match else "REJECTED"
        qid_str = f"query={query_id[:8]} " if query_id else ""
        ctx_str = f"[{context_info}] " if context_info else ""
        dino_str = f"dino={dino_sim:.3f} " if dino_sim is not None else ""
        print(
            f"[target-filter] {status_tag} {ctx_str}{qid_str}class={query_target.get('type', 'vehicle')} "
            f"score={eval_score:.3f} (thresh={target_thresh:.2f}) "
            f"{dino_str}color={color_sim:.3f} (thresh={VEHICLE_COLOR_THRESHOLD:.2f}) "
            f"plate={plate_status} (cand='{plate_text}', query='{query_plate}') "
            f"reason='{decision_reason}'"
        )

        return is_match, plate_text, plate_status, color_sim, decision_reason

    def extract_targets(self, frame, det_thresh=0.45, min_crop=30, extract_plates=True, is_query=False):
        """
        Extracts all targets (person faces + vehicle crops with embeddings) from a single frame.
        Thread-safe with periodic memory management for Jetson Nano.
        When is_query=True, prioritizes central/prominent subjects, suppresses overlapping
        duplicate boxes (IoU NMS), and falls back to whole-crop if YOLO misses tightly cropped vehicle.
        """
        with self._infer_lock:
            targets = []

            # 1. Face detection & embeddings
            for face in self.face_app.get(frame):
                if face.det_score > det_thresh:
                    emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-6)
                    bbox = face.bbox.astype(int).tolist()
                    targets.append({
                        "type": "person",
                        "emb": emb,
                        "bbox": bbox,
                        "thresh": self.face_threshold,
                        "conf": float(face.det_score),
                    })

            # 2. YOLO vehicle detection & hybrid Re-ID
            fh, fw = frame.shape[:2]
            vehicle_raw = []
            if self.dinov2 is not None:
                results = self.yolo(frame, verbose=False)[0]
                conf_min = (0.15 if is_query else VEHICLE_CONF_MIN)
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in VEHICLE_YOLO_CLASSES:
                        conf = float(box.conf[0])
                        if conf < conf_min:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(fw, x2), min(fh, y2)
                        if (x2 - x1) > min_crop and (y2 - y1) > min_crop:
                            vehicle_raw.append({
                                "cls_id": cls_id,
                                "type": VEHICLE_YOLO_CLASSES[cls_id],
                                "conf": conf,
                                "bbox": [x1, y1, x2, y2],
                            })

                # IoU suppression for duplicate/overlapping boxes of compatible type
                filtered_vehicles = []
                for v in sorted(vehicle_raw, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=True):
                    b1 = v["bbox"]
                    is_dup = False
                    for kept in filtered_vehicles:
                        if is_compatible_target_type(v["type"], kept["type"]):
                            b2 = kept["bbox"]
                            inter = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                            iou = inter / float(a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0
                            containment = inter / float(min(a1, a2)) if min(a1, a2) > 0 else 0.0
                            if iou > 0.40 or containment > 0.70:
                                is_dup = True
                                break
                    if not is_dup:
                        filtered_vehicles.append(v)

                # For query images, sort by prominence (area + centrality)
                if is_query and filtered_vehicles:
                    def _prominence(v):
                        b = v["bbox"]
                        area = (b[2] - b[0]) * (b[3] - b[1])
                        cx = (b[0] + b[2]) / 2.0
                        cy = (b[1] + b[3]) / 2.0
                        dist = np.hypot(cx - fw / 2.0, cy - fh / 2.0)
                        diag = np.hypot(fw, fh) + 1e-6
                        return (area / (1.0 + 1.5 * (dist / diag))) * (0.5 + 0.5 * v["conf"])
                    filtered_vehicles.sort(key=_prominence, reverse=True)

                # Fallback: if is_query and no vehicles/faces detected, but frame is large enough, treat whole frame as vehicle crop
                if is_query and not filtered_vehicles and not targets and fw >= min_crop and fh >= min_crop:
                    filtered_vehicles.append({
                        "cls_id": 2,
                        "type": "car",
                        "conf": 1.0,
                        "bbox": [0, 0, fw, fh],
                    })

                for v in filtered_vehicles:
                    x1, y1, x2, y2 = v["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] > min_crop and crop.shape[1] > min_crop:
                        emb = self.extract_vehicle_hybrid_embedding(crop)
                        color_sig = self.extract_vehicle_color_signature(crop)
                        plate_text = ""
                        if extract_plates and self.enable_lpr:
                            plate_text = self.plate_verifier.extract_plate_text(crop)
                        targets.append({
                            "type": v["type"],
                            "emb": emb,
                            "color_sig": color_sig,
                            "crop": crop,
                            "bbox": [x1, y1, x2, y2],
                            "thresh": self.vehicle_threshold,
                            "conf": v["conf"],
                            "plate_text": plate_text,
                        })

            self._infer_count += 1
            if self._infer_count % 40 == 0:
                gc.collect()
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()

            return targets


def _target_summary(targets):
    """Summarizes target types count (e.g. 'person:1, car:1')."""
    counts = Counter(t["type"] for t in targets)
    return ", ".join(f"{k}:{v}" for k, v in counts.items()) or "none detected"


def _print_target_breakdown(label, targets):
    print(f"[{label}] targets -> {_target_summary(targets)}")
