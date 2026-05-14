"""
╔══════════════════════════════════════════════════════════════════════╗
║   FINGERPRINT ATTENDANCE & ACCESS CONTROL — CUSTOM DATASET          ║
║   Biometrics Project — Scenario Evaluation                          ║
║                                                                      ║
║   Uses YOUR OWN fingerprint dataset (built with collect_dataset.py) ║
║   Reuses GaborMethod + preprocess from main.py (unchanged)          ║
║                                                                      ║
║   Input  : Live fingerprint image from phone/webcam camera          ║
║   Output : GRANTED → logs attendance | DENIED → access blocked     ║
║                                                                      ║
║   SETUP:                                                             ║
║     1. Run collect_dataset.py first (builds my_dataset/)            ║
║     2. Set PHONE_IP below (same as collect_dataset.py)              ║
║     3. Run: python attendance_app_custom.py                         ║
║                                                                      ║
║   CONTROLS (live window):                                            ║
║     SPACE  → capture & identify                                     ║
║     R      → re-enroll (rebuild gallery)                            ║
║     L      → print attendance log to console                        ║
║     Q      → quit                                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import csv
import time
import datetime
import numpy as np
import cv2

# ── Import from your existing project (main.py must be in same folder) ──
try:
    from main import GaborMethod, preprocess
except ImportError:
    print("[ERROR] Could not import from main.py.")
    print("        Place attendance_app_custom.py in the same folder as main.py.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
PHONE_IP          = "192.168.1.5"      # ← same IP as collect_dataset.py
PHONE_PORT        = 8080
DATASET_DIR       = "my_dataset"       # folder built by collect_dataset.py
LOG_FILE          = "attendance_log.csv"
MATCH_THRESHOLD   = 0.55               # cosine similarity cutoff (lower than
                                       # SOCOFing because phone camera images
                                       # have more variability — tune this!)
TRAIN_SPLIT       = 0.7                # 70% enroll, 30% kept for testing
# ─────────────────────────────────────────────────────────────────

STREAM_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"

# Colors (BGR)
GREEN  = ( 60, 200,  80)
RED    = ( 50,  50, 220)
BLUE   = (220, 130,  50)
YELLOW = ( 30, 210, 240)
WHITE  = (255, 255, 255)
DARK   = ( 18,  18,  18)


# ══════════════════════════════════════════════════════════════════
# 1. CUSTOM DATASET LOADER
# ══════════════════════════════════════════════════════════════════

def load_custom_dataset(dataset_dir: str):
    """
    Loads images from:
      my_dataset/
        Ahmed/  → ahmed_001.png, ahmed_002.png, ...
        Omar/   → omar_001.png, ...

    Returns:
        dataset: { subject_name: [img_path, ...] }
    """
    dataset = {}
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        print("        Run collect_dataset.py first.")
        sys.exit(1)

    for subject in sorted(os.listdir(dataset_dir)):
        subject_dir = os.path.join(dataset_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        images = sorted([
            os.path.join(subject_dir, f)
            for f in os.listdir(subject_dir)
            if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))
        ])
        if len(images) >= 2:   # need at least 2 images
            dataset[subject] = images
        else:
            print(f"[WARN] {subject} has only {len(images)} image(s) — skipping.")

    if not dataset:
        print(f"[ERROR] No valid subjects found in {dataset_dir}.")
        print("        Run collect_dataset.py to build your dataset.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(dataset)} subjects:")
    for name, imgs in dataset.items():
        print(f"       {name:<20} {len(imgs)} images")
    return dataset


def split_custom_dataset(dataset: dict, train_ratio: float = 0.7):
    """Split each subject's images into train (enroll) and test (probe) sets."""
    train, test = {}, {}
    for sid, paths in dataset.items():
        n_train = max(1, int(len(paths) * train_ratio))
        train[sid] = paths[:n_train]
        test[sid]  = paths[n_train:] if len(paths) > n_train else [paths[-1]]
    return train, test


# ══════════════════════════════════════════════════════════════════
# 2. CUSTOM PREPROCESS  (adapted for phone camera images)
# ══════════════════════════════════════════════════════════════════

def preprocess_phone_image(img_path_or_array, size=(128, 128)):
    """
    Enhanced preprocessing for phone camera images.
    Handles both file paths and numpy arrays (for live frames).

    Steps:
      1. Load / convert to grayscale
      2. Crop center region (fingertip pad)
      3. Resize to 128×128
      4. CLAHE with stronger clip (phone images have less contrast than scanners)
      5. Adaptive thresholding to emphasize ridges
      6. Gaussian blur to reduce noise
    """
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_path_or_array}")
    else:
        # numpy array (BGR frame from camera)
        img = cv2.cvtColor(img_path_or_array, cv2.COLOR_BGR2GRAY)

    # If image is much larger than 128×128, crop center first
    h, w = img.shape
    if h > 300 and w > 300:
        # Crop central 60% of the image (finger pad region)
        cy, cx = h // 2, w // 2
        crop_h, crop_w = int(h * 0.6), int(w * 0.6)
        img = img[cy - crop_h//2 : cy + crop_h//2,
                  cx - crop_w//2 : cx + crop_w//2]

    # Resize
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # CLAHE — stronger than main.py's clipLimit=2.0 for phone images
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ══════════════════════════════════════════════════════════════════
# 3. CUSTOM GABOR METHOD (wraps main.py's GaborMethod with our preprocessing)
# ══════════════════════════════════════════════════════════════════

class CustomGaborMethod(GaborMethod):
    """
    Extends GaborMethod from main.py.
    Overrides enroll and identify to use preprocess_phone_image()
    instead of main.py's preprocess() — better suited for phone camera.
    Everything else (kernels, cosine similarity, gallery) is inherited.
    """

    def enroll(self, train_set: dict):
        """Build gallery from phone-captured training images."""
        print("[ENROLL] Building fingerprint gallery...")
        for sid, paths in train_set.items():
            vecs = []
            for p in paths:
                try:
                    img = preprocess_phone_image(p)
                    vecs.append(self.extract(img))
                except Exception as e:
                    print(f"[WARN] Skipping {p}: {e}")
            if vecs:
                mean_vec = np.mean(vecs, axis=0)
                self.gallery[sid] = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
                print(f"  ✓ Enrolled: {sid} ({len(vecs)} images)")
        print(f"[ENROLL] Gallery ready — {len(self.gallery)} subjects enrolled.\n")

    def identify_frame(self, frame: np.ndarray):
        """
        Identify from a live camera frame (numpy BGR array).
        Returns ranked list of (subject_id, score).
        """
        img = preprocess_phone_image(frame)
        probe_vec = self.extract(img)
        probe_vec = probe_vec / (np.linalg.norm(probe_vec) + 1e-8)
        scores = {sid: self.cosine_sim(probe_vec, gvec)
                  for sid, gvec in self.gallery.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def identify(self, probe_path: str):
        """Override to use phone preprocessing for file paths too."""
        img = preprocess_phone_image(probe_path)
        probe_vec = self.extract(img)
        probe_vec = probe_vec / (np.linalg.norm(probe_vec) + 1e-8)
        scores = {sid: self.cosine_sim(probe_vec, gvec)
                  for sid, gvec in self.gallery.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════════
# 4. ATTENDANCE LOG
# ══════════════════════════════════════════════════════════════════

def init_log(log_file: str):
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Identity", "Score", "Result"])


def log_event(log_file: str, identity: str, score: float, result: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, identity, f"{score:.4f}", result])
    emoji = "✓" if result == "GRANTED" else "✗"
    print(f"  [{result}] {emoji} {identity:<20} score={score:.4f}  @ {ts}")


def print_log(log_file: str):
    if not os.path.exists(log_file):
        print("[LOG] No attendance log yet.")
        return
    print(f"\n{'═'*65}")
    print(f"  ATTENDANCE LOG — {log_file}")
    print(f"{'═'*65}")
    with open(log_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)
    if not rows:
        print("  (empty)")
    for row in rows[-20:]:   # show last 20
        ts, identity, score, result = row
        marker = "✓" if result == "GRANTED" else "✗"
        print(f"  {marker} {ts}  {identity:<20} {score}  {result}")
    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════
# 5. UI OVERLAY
# ══════════════════════════════════════════════════════════════════

def draw_attendance_ui(frame, enrolled_names, last_result, last_identity,
                        last_score, scanning=False):
    """
    Full attendance HUD overlay.
    """
    h, w = frame.shape[:2]

    # ── Fingertip guide box ──────────────────────────────────────
    box_w, box_h = 260, 340
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_w//2, cy - box_h//2
    x2, y2 = cx + box_w//2, cy + box_h//2

    if last_result is None:
        guide_color = BLUE if not scanning else YELLOW
    elif last_result:
        guide_color = GREEN
    else:
        guide_color = RED

    # Corner brackets
    L, T = 35, 3
    for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px,py), (px+dx*L, py), guide_color, T)
        cv2.line(frame, (px,py), (px, py+dy*L), guide_color, T)

    # Darken outside guide box
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (255,255,255), -1)
    alpha_mask = overlay[:,:,0] == 0
    frame[alpha_mask] = (frame[alpha_mask] * 0.5).astype(np.uint8)

    # Scanning pulse effect
    if scanning:
        pulse_y = y1 + int((time.time() % 1.0) * box_h)
        cv2.line(frame, (x1, pulse_y), (x2, pulse_y), YELLOW, 2)

    # ── TOP BANNER ───────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w,65), (12,12,12), -1)

    # Title
    cv2.putText(frame, "FINGERPRINT ATTENDANCE SYSTEM",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    # Enrolled count badge
    badge_text = f"{len(enrolled_names)} enrolled"
    cv2.rectangle(frame, (w-130, 10), (w-10, 50), (40,40,40), -1)
    cv2.putText(frame, badge_text, (w-125, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

    # ── RESULT PANEL (bottom) ────────────────────────────────────
    panel_h = 120
    panel_color = (12,12,12)
    if last_result is True:
        panel_color = (20, 60, 20)
    elif last_result is False:
        panel_color = (20, 20, 60)

    cv2.rectangle(frame, (0, h-panel_h), (w, h), panel_color, -1)

    if scanning:
        status_text = "Analyzing fingerprint..."
        cv2.putText(frame, status_text, (15, h-panel_h+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, YELLOW, 2, cv2.LINE_AA)

    elif last_result is None:
        cv2.putText(frame, "Place finger in box, press SPACE",
                    (15, h-panel_h+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 1, cv2.LINE_AA)

    elif last_result is True:
        # GRANTED
        cv2.putText(frame, f"ACCESS GRANTED",
                    (15, h-panel_h+38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 3, cv2.LINE_AA)
        cv2.putText(frame, f"Welcome, {last_identity}!",
                    (15, h-panel_h+68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {last_score:.4f}  |  Threshold: {MATCH_THRESHOLD}",
                    (15, h-panel_h+92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)

    else:
        # DENIED
        cv2.putText(frame, f"ACCESS DENIED",
                    (15, h-panel_h+38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, RED, 3, cv2.LINE_AA)
        cv2.putText(frame, "Fingerprint not recognized.",
                    (15, h-panel_h+68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Best score: {last_score:.4f}  |  Required: {MATCH_THRESHOLD}",
                    (15, h-panel_h+92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "SPACE=Scan  R=Re-enroll  L=Log  Q=Quit",
                (15, h-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1, cv2.LINE_AA)

    # ── Enrolled names sidebar ───────────────────────────────────
    sidebar_x = 10
    cv2.putText(frame, "Enrolled:", (sidebar_x, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1, cv2.LINE_AA)
    for i, name in enumerate(enrolled_names):
        dot_color = GREEN if (last_result and last_identity == name) else (100,100,100)
        cv2.circle(frame, (sidebar_x+6, y1+42+i*22), 4, dot_color, -1)
        name_color = GREEN if (last_result and last_identity == name) else WHITE
        cv2.putText(frame, name, (sidebar_x+16, y1+48+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, name_color, 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════
# 6. QUICK ACCURACY TEST  (runs over test split before live mode)
# ══════════════════════════════════════════════════════════════════

def run_accuracy_test(gabor, test_set):
    """
    Tests the model on held-out images before going live.
    Prints a simple confusion report.
    """
    print("\n" + "═"*55)
    print("  PRE-SESSION ACCURACY TEST (held-out test images)")
    print("═"*55)

    correct = 0
    total   = 0
    results = []

    for true_id, paths in test_set.items():
        for img_path in paths:
            ranked = gabor.identify(img_path)
            if not ranked:
                continue
            pred_id, score = ranked[0]
            granted = score >= MATCH_THRESHOLD
            is_correct = (pred_id == true_id) and granted
            correct += is_correct
            total   += 1
            results.append((true_id, pred_id, score, granted, is_correct))

    for true_id, pred_id, score, granted, is_correct in results:
        marker = "✓" if is_correct else "✗"
        result_str = "GRANTED" if granted else "DENIED"
        print(f"  {marker} True:{true_id:<15} Pred:{pred_id:<15} "
              f"Score:{score:.4f} → {result_str}")

    acc = (correct / total * 100) if total > 0 else 0
    print(f"\n  Rank-1 Accuracy : {correct}/{total} = {acc:.1f}%")
    print(f"  Threshold used  : {MATCH_THRESHOLD}")
    if acc < 50:
        print(f"\n  [TIP] Accuracy is low. Try:")
        print(f"        • Capturing more images per subject (>10)")
        print(f"        • Better lighting when scanning")
        print(f"        • Lowering MATCH_THRESHOLD (currently {MATCH_THRESHOLD})")
    print("═"*55 + "\n")
    return acc


# ══════════════════════════════════════════════════════════════════
# 7. LIVE CAMERA SESSION
# ══════════════════════════════════════════════════════════════════

def run_live_session(gabor, cap, log_file):
    """
    Main live attendance loop.
    """
    enrolled_names = list(gabor.gallery.keys())
    last_result    = None
    last_identity  = ""
    last_score     = 0.0
    result_expiry  = 0       # timestamp when to clear result display
    RESULT_HOLD    = 3.0     # seconds to hold result on screen

    tmp_path = "_tmp_live_capture.png"

    print("\n[LIVE] Attendance system active.")
    print(f"[LIVE] Enrolled: {', '.join(enrolled_names)}")
    print(f"[LIVE] Threshold: {MATCH_THRESHOLD}")
    print("[LIVE] Press SPACE to scan, Q to quit.\n")

    scanning = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            time.sleep(0.05)
            continue

        now = time.time()

        # Clear result after hold time
        if last_result is not None and now > result_expiry:
            last_result = None
            last_identity = ""
            last_score = 0.0

        display = draw_attendance_ui(
            frame.copy(), enrolled_names,
            last_result, last_identity, last_score,
            scanning=scanning
        )
        cv2.imshow("Fingerprint Attendance System", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord('l') or key == ord('L'):
            print_log(log_file)

        elif key == ord('r') or key == ord('R'):
            print("[INFO] Re-enrollment requested — restart the app.")
            break

        elif key == ord(' ') and not scanning:
            scanning = True

            # Show scanning frame
            scan_display = draw_attendance_ui(
                frame.copy(), enrolled_names,
                None, "", 0.0, scanning=True
            )
            cv2.imshow("Fingerprint Attendance System", scan_display)
            cv2.waitKey(1)

            # Save frame and identify
            cv2.imwrite(tmp_path, frame)
            ranked = gabor.identify_frame(frame)

            scanning = False

            if ranked:
                pred_id, score = ranked[0]
                granted = score >= MATCH_THRESHOLD
                last_result   = granted
                last_identity = pred_id if granted else "Unknown"
                last_score    = score
                result_expiry = now + RESULT_HOLD
                log_event(log_file, last_identity, score,
                          "GRANTED" if granted else "DENIED")
            else:
                last_result   = False
                last_identity = "Unknown"
                last_score    = 0.0
                result_expiry = now + RESULT_HOLD

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FINGERPRINT ATTENDANCE SYSTEM — CUSTOM DATASET")
    print("=" * 60)

    # ── Load custom dataset ──────────────────────────────────────
    print(f"\n[STEP 1] Loading dataset from: {DATASET_DIR}/")
    dataset = load_custom_dataset(DATASET_DIR)
    train_set, test_set = split_custom_dataset(dataset, TRAIN_SPLIT)

    n_train = sum(len(v) for v in train_set.values())
    n_test  = sum(len(v) for v in test_set.values())
    print(f"\n[INFO] Train (enroll): {n_train} images | Test (probe): {n_test} images")

    # ── Enroll gallery ───────────────────────────────────────────
    print(f"\n[STEP 2] Enrolling subjects with Gabor features...")
    gabor = CustomGaborMethod(num_orientations=8, num_scales=4)
    gabor.enroll(train_set)

    # ── Quick accuracy test ──────────────────────────────────────
    print("[STEP 3] Running accuracy test on held-out images...")
    acc = run_accuracy_test(gabor, test_set)

    # ── Init log ─────────────────────────────────────────────────
    init_log(LOG_FILE)

    # ── Connect camera ───────────────────────────────────────────
    print(f"[STEP 4] Connecting to camera at {STREAM_URL} ...")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print(f"[WARN] Phone camera not reachable. Trying laptop webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No camera available. Exiting.")
            sys.exit(1)
        print("[INFO] Using laptop webcam.")
    else:
        print("[OK] Phone camera connected!")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # ── Live attendance session ──────────────────────────────────
    print(f"\n[STEP 5] Starting live attendance session...")
    run_live_session(gabor, cap, LOG_FILE)

    cap.release()
    print(f"\n[DONE] Session ended. Log saved to: {LOG_FILE}")
    print_log(LOG_FILE)


if __name__ == "__main__":
    main()
