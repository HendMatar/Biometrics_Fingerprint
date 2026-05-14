"""
╔══════════════════════════════════════════════════════════════════╗
║   FINGERPRINT ATTENDANCE & ACCESS CONTROL SYSTEM                ║
║   Scenario Evaluation — Real-Life Application                   ║
║                                                                  ║
║   Builds on top of main.py (unchanged) — reuses:               ║
║     • GaborMethod   (feature extraction + matching)             ║
║     • preprocess    (CLAHE + Gaussian preprocessing)            ║
║     • load_dataset  (SOCOFing loader)                           ║
║     • split_dataset (80/20 per-subject split)                   ║
║                                                                  ║
║   Scenario:                                                      ║
║     Input  → Fingerprint image captured from camera             ║
║     Output → GRANTED (log attendance) | DENIED (alarm)         ║
║                                                                  ║
║   How to run:                                                    ║
║     python attendance_app.py                                     ║
║                                                                  ║
║   Controls (live camera window):                                 ║
║     SPACE  → capture & identify current frame                   ║
║     Q      → quit                                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import csv
import datetime
import cv2
import numpy as np

# ── Import everything we need from the existing project ─────────
# main.py must be in the same directory as this file.
try:
    from main import GaborMethod, preprocess, load_dataset, split_dataset
except ImportError:
    print("[ERROR] Could not import from main.py.")
    print("        Make sure attendance_app.py is in the same folder as main.py.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  (mirrors main.py — adjust if needed)
# ─────────────────────────────────────────────────────────────────
DATASET_ROOT      = r".\SOCOFing"   # same as main.py
NUM_SUBJECTS      = 100              # subjects to enroll
IMAGES_PER_SUBJECT = 10
TRAIN_RATIO       = 0.8

# Access-control threshold (cosine similarity, 0–1).
# Gabor scores for genuine pairs cluster near 0.85–0.99.
# Tune this if you get too many false accepts/rejects.
MATCH_THRESHOLD   = 0.80

# Attendance log file
LOG_FILE          = "attendance_log.csv"

# Camera index (0 = default webcam)
CAMERA_INDEX      = 0

# ─────────────────────────────────────────────────────────────────
# FRIENDLY DISPLAY NAMES
# Extracts "Subject 42 — Left index" from SOCOFing filename style:
#   "42__M_Left_index_finger"
# ─────────────────────────────────────────────────────────────────
def friendly_name(finger_id: str) -> str:
    parts = finger_id.split("__")
    subject_num = parts[0].strip()
    if len(parts) > 1:
        rest = parts[1].replace("_finger", "").replace("_", " ").strip()
        # strip gender token (single letter)
        tokens = rest.split()
        tokens = [t for t in tokens if len(t) > 1]
        description = " ".join(tokens)
        return f"Subject {subject_num} — {description}"
    return f"Subject {subject_num}"


# ─────────────────────────────────────────────────────────────────
# ATTENDANCE LOGGER
# ─────────────────────────────────────────────────────────────────
def init_log(log_file: str):
    """Create log CSV with headers if it doesn't exist yet."""
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Identity", "Score", "Result"])
        print(f"[LOG] Created attendance log: {log_file}")


def log_event(log_file: str, identity: str, score: float, result: str):
    """Append one attendance/access event to the CSV log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, identity, f"{score:.4f}", result])
    print(f"[LOG] {timestamp} | {identity} | score={score:.4f} | {result}")


# ─────────────────────────────────────────────────────────────────
# IDENTIFY FROM IMAGE PATH  (used by both camera + file demo modes)
# ─────────────────────────────────────────────────────────────────
def identify_fingerprint(gabor: GaborMethod, img_path: str, threshold: float):
    """
    Runs Gabor identification on a saved image file.

    Returns:
        identity  (str)   — matched subject ID or "UNKNOWN"
        score     (float) — top cosine similarity score
        granted   (bool)  — True if score ≥ threshold
    """
    ranked = gabor.identify(img_path)
    if not ranked:
        return "UNKNOWN", 0.0, False

    top_id, top_score = ranked[0]
    granted = top_score >= threshold
    identity = friendly_name(top_id) if granted else "UNKNOWN"
    return identity, float(top_score), granted


# ─────────────────────────────────────────────────────────────────
# OVERLAY RENDERER  (draws the HUD on the camera frame)
# ─────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, status: str, identity: str,
             score: float, granted: bool | None) -> np.ndarray:
    """
    Draws a semi-transparent HUD panel at the bottom of the frame.
    Color coding:
      • Blue  → waiting / scanning
      • Green → GRANTED
      • Red   → DENIED
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel background
    panel_h = 110
    if granted is None:
        color = (180, 100, 20)      # blue-ish (BGR)
    elif granted:
        color = (30, 140, 30)       # green
    else:
        color = (30, 30, 180)       # red

    cv2.rectangle(overlay, (0, h - panel_h), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    big_font  = 1.0
    small_font = 0.55
    white     = (255, 255, 255)
    y_base    = h - panel_h + 30

    # Status line
    cv2.putText(frame, status, (12, y_base),
                font, big_font, white, 2, cv2.LINE_AA)

    # Identity + score lines
    if granted is not None:
        id_text    = f"Identity : {identity}"
        score_text = f"Score    : {score:.4f}   (threshold >= {MATCH_THRESHOLD})"
        cv2.putText(frame, id_text,    (12, y_base + 32), font, small_font, white, 1, cv2.LINE_AA)
        cv2.putText(frame, score_text, (12, y_base + 58), font, small_font, white, 1, cv2.LINE_AA)

    # Crosshair / capture hint
    cx, cy = w // 2, (h - panel_h) // 2
    cv2.rectangle(frame, (cx - 80, cy - 100), (cx + 80, cy + 100), (200, 200, 200), 1)
    hint = "SPACE=Capture   Q=Quit"
    cv2.putText(frame, hint, (12, h - panel_h - 8),
                font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────
# DEMO MODE  (no camera — iterates over test-set images)
# Useful on machines without a webcam or fingerprint scanner.
# ─────────────────────────────────────────────────────────────────
def run_demo_mode(gabor: GaborMethod, test_set: dict, log_file: str):
    """
    Simulates the attendance system by running it over the test set.
    Prints a colour-coded console table and writes the CSV log.
    """
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    print(f"\n{'═'*65}")
    print(f"  DEMO MODE — Simulated Attendance Check ({len(test_set)} subjects)")
    print(f"{'═'*65}")
    print(f"  {'IDENTITY':<35} {'SCORE':>7}  RESULT")
    print(f"  {'─'*35} {'─'*7}  {'─'*10}")

    granted_count = 0
    denied_count  = 0

    for sid, paths in list(test_set.items())[:30]:   # show first 30 for brevity
        for img_path in paths[:1]:                   # one probe per subject
            identity, score, granted = identify_fingerprint(
                gabor, img_path, MATCH_THRESHOLD
            )
            result_str = "GRANTED ✓" if granted else "DENIED  ✗"
            color      = GREEN if granted else RED
            name       = friendly_name(sid)

            print(f"  {color}{name:<35} {score:>7.4f}  {result_str}{RESET}")
            log_event(log_file, identity, score, "GRANTED" if granted else "DENIED")

            if granted:
                granted_count += 1
            else:
                denied_count += 1

    print(f"\n{'═'*65}")
    print(f"  {BOLD}Summary{RESET} — Granted: {GREEN}{granted_count}{RESET} | "
          f"Denied: {RED}{denied_count}{RESET}")
    print(f"  Log saved → {log_file}")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────
# LIVE CAMERA MODE
# ─────────────────────────────────────────────────────────────────
def run_camera_mode(gabor: GaborMethod, log_file: str):
    """
    Opens the webcam.  Press SPACE to capture the current frame,
    save it as a temp file, run Gabor identification, and display
    the access-control result with a colour overlay.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        print("        Switch to demo mode: python attendance_app.py --demo")
        return

    print("\n[CAMERA] Live fingerprint scanner active.")
    print("         Place finger on scanner, press SPACE to capture.\n")

    tmp_path = "_tmp_fingerprint_capture.png"
    status   = "Place finger & press SPACE"
    identity = ""
    score    = 0.0
    granted  = None          # None = no scan yet

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        display = draw_hud(frame.copy(), status, identity, score, granted)
        cv2.imshow("Fingerprint Attendance System", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            # ── Capture & identify ──────────────────────────
            cv2.imwrite(tmp_path, frame)
            status  = "Scanning…"
            granted = None
            display = draw_hud(frame.copy(), status, identity, score, granted)
            cv2.imshow("Fingerprint Attendance System", display)
            cv2.waitKey(1)

            identity, score, granted = identify_fingerprint(
                gabor, tmp_path, MATCH_THRESHOLD
            )

            if granted:
                status = f"✓ ACCESS GRANTED"
                print(f"\n  [GRANTED] {identity}  (score={score:.4f})")
            else:
                status = "✗ ACCESS DENIED — Unknown fingerprint"
                print(f"\n  [DENIED]  Unknown  (score={score:.4f})")

            log_event(log_file, identity, score, "GRANTED" if granted else "DENIED")

            # Hold result on screen for 2 s
            result_frame = draw_hud(frame.copy(), status, identity, score, granted)
            cv2.imshow("Fingerprint Attendance System", result_frame)
            cv2.waitKey(2000)

            status = "Place finger & press SPACE"
            granted = None

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"\n[LOG] Session log saved → {log_file}")


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main():
    demo_mode = "--demo" in sys.argv

    # ── Step 1: Load & split dataset (same pipeline as main.py) ──
    print("\n[STEP 1] Loading SOCOFing dataset...")
    dataset = load_dataset(DATASET_ROOT, NUM_SUBJECTS, IMAGES_PER_SUBJECT)
    train_set, test_set = split_dataset(dataset, TRAIN_RATIO)
    print(f"[INFO]   Enrolled subjects : {len(train_set)}")
    print(f"[INFO]   Test probes       : {sum(len(v) for v in test_set.values())}")

    # ── Step 2: Enroll (build gallery with Gabor features) ───────
    print("\n[STEP 2] Building enrollment gallery (Gabor features)…")
    gabor = GaborMethod(num_orientations=8, num_scales=4)
    gabor.enroll(train_set)
    print(f"[INFO]   Gallery size: {len(gabor.gallery)} subjects enrolled.")

    # ── Step 3: Init attendance log ───────────────────────────────
    init_log(LOG_FILE)

    # ── Step 4: Run application ───────────────────────────────────
    if demo_mode:
        print("\n[MODE] Demo — running over test set images (no camera required).")
        run_demo_mode(gabor, test_set, LOG_FILE)
    else:
        print("\n[MODE] Live camera mode. Use --demo flag for no-camera simulation.")
        run_camera_mode(gabor, LOG_FILE)


if __name__ == "__main__":
    main()
