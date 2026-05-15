"""
Fingerprint Attendance & Access Control System
(uses existing main.py without modifying it)

Run:
    python attendance_app.py
or:
    python attendance_app.py --demo
"""

import os
import sys
import csv
import datetime
import tempfile
import cv2
import numpy as np

# -------------------------------------------------
# IMPORT FROM main.py
# -------------------------------------------------
try:
    from main import GaborMethod, load_dataset, split_dataset
except ImportError:
    print("[ERROR] main.py not found in same folder.")
    sys.exit(1)


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_ROOT = r".\SOCOFing"
NUM_SUBJECTS = 100
IMAGES_PER_SUBJECT = 10
TRAIN_RATIO = 0.8


MATCH_THRESHOLD = 0.995

LOG_FILE = "attendance_log.csv"
CAMERA_INDEX = 0


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def friendly_name(finger_id):
    parts = finger_id.split("__")
    subject_num = parts[0]

    if len(parts) > 1:
        rest = parts[1].replace("_finger", "")
        rest = rest.replace("_", " ")
        tokens = [t for t in rest.split() if len(t) > 1]
        return f"Subject {subject_num} — {' '.join(tokens)}"

    return f"Subject {subject_num}"


def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Identity", "Score", "Result"])


def log_event(identity, score, result):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, identity, f"{score:.4f}", result])

    print(
        f"[LOG] {timestamp} | {identity} | "
        f"{score:.4f} | {result}"
    )


# -------------------------------------------------
# IMAGE QUALITY CHECK
# -------------------------------------------------
def is_good_quality(gray):
    """
    Reject very blurry or blank images
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap < 50:
        return False

    return True


# -------------------------------------------------
# EXTRACT CENTER ROI
# -------------------------------------------------
def extract_fingerprint_roi(frame):
    """
    Crop center of frame only.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    x1 = w // 4
    x2 = 3 * w // 4
    y1 = h // 4
    y2 = 3 * h // 4

    roi = gray[y1:y2, x1:x2]

    roi = cv2.resize(roi, (128, 128))

    return roi


# -------------------------------------------------
# IDENTIFY FROM NUMPY IMAGE
# -------------------------------------------------
def identify_from_array(gabor, img_array):
    """
    Saves temp file because main.GaborMethod expects path.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as tmp:

        tmp_path = tmp.name

    cv2.imwrite(tmp_path, img_array)

    try:
        ranked = gabor.identify(tmp_path)

        if not ranked:
            return "UNKNOWN", 0.0, False

        sid, score = ranked[0]

        if score >= MATCH_THRESHOLD:
            return friendly_name(sid), score, True
        else:
            return "UNKNOWN", score, False

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------------------------------------------------
# DRAW HUD
# -------------------------------------------------
def draw_hud(frame, text, color):
    out = frame.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 60), color, -1)

    cv2.putText(
        out,
        text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    h, w = out.shape[:2]

    cv2.rectangle(
        out,
        (w//4, h//4),
        (3*w//4, 3*h//4),
        (255, 255, 255),
        2
    )

    return out


# -------------------------------------------------
# DEMO MODE
# -------------------------------------------------
def run_demo(gabor, test_set):
    print("\n[DEMO MODE]\n")

    for sid, paths in list(test_set.items())[:20]:

        img = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)

        identity, score, granted = identify_from_array(
            gabor,
            img
        )

        result = "GRANTED" if granted else "DENIED"

        print(identity, score, result)

        log_event(identity, score, result)


# -------------------------------------------------
# LIVE MODE
# -------------------------------------------------
def run_camera(gabor):
    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[CAMERA ACTIVE]")
    print("SPACE = capture")
    print("Q     = quit\n")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] camera read failed")
                break

            display = draw_hud(
                frame,
                "Place finger in box | SPACE capture | Q quit",
                (120, 80, 20)
            )

            cv2.imshow(
                "Fingerprint Attendance",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):

                roi = extract_fingerprint_roi(frame)

                if not is_good_quality(roi):
                    print("[DENIED] poor quality image")

                    fail = draw_hud(
                        frame,
                        "Poor quality - try again",
                        (0, 0, 255)
                    )

                    cv2.imshow(
                        "Fingerprint Attendance",
                        fail
                    )

                    cv2.waitKey(1500)
                    continue

                identity, score, granted = identify_from_array(
                    gabor,
                    roi
                )

                if granted:
                    print(
                        f"[GRANTED] {identity} "
                        f"({score:.4f})"
                    )

                    log_event(
                        identity,
                        score,
                        "GRANTED"
                    )

                    result = draw_hud(
                        frame,
                        f"ACCESS GRANTED - {identity}",
                        (0, 180, 0)
                    )

                else:
                    print(
                        f"[DENIED] UNKNOWN "
                        f"({score:.4f})"
                    )

                    log_event(
                        identity,
                        score,
                        "DENIED"
                    )

                    result = draw_hud(
                        frame,
                        "ACCESS DENIED",
                        (0, 0, 255)
                    )

                cv2.imshow(
                    "Fingerprint Attendance",
                    result
                )

                # responsive delay
                for _ in range(200):
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        return

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nClosed cleanly.")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    demo_mode = "--demo" in sys.argv

    print("\n[STEP 1] Loading dataset...")
    dataset = load_dataset(
        DATASET_ROOT,
        NUM_SUBJECTS,
        IMAGES_PER_SUBJECT
    )

    train_set, test_set = split_dataset(
        dataset,
        TRAIN_RATIO
    )

    print("[STEP 2] Building Gabor gallery...")
    gabor = GaborMethod()
    gabor.enroll(train_set)

    print(
        f"[INFO] Enrolled {len(gabor.gallery)} subjects"
    )

    init_log()

    if demo_mode:
        run_demo(gabor, test_set)
    else:
        run_camera(gabor)


if __name__ == "__main__":
    main()