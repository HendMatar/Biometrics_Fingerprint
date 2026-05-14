"""
╔══════════════════════════════════════════════════════════════════════╗
║   FINGERPRINT DATASET COLLECTOR                                      ║
║   Biometrics Project — Custom Dataset Builder                        ║
║                                                                      ║
║   Uses your Android phone as a high-resolution camera via            ║
║   IP Webcam app (free on Play Store).                                ║
║                                                                      ║
║   SETUP (do this once):                                              ║
║     1. Install "IP Webcam" by Pavel Khlebovich from Play Store       ║
║     2. Open IP Webcam → scroll down → tap "Start server"            ║
║     3. Note the IP shown on screen, e.g. http://192.168.1.5:8080    ║
║     4. Set PHONE_IP below to match                                   ║
║     5. Make sure phone & PC are on the same Wi-Fi network           ║
║                                                                      ║
║   HOW TO USE:                                                        ║
║     python collect_dataset.py                                        ║
║     → Enter subject name when prompted                               ║
║     → Hold finger steady in the green box                           ║
║     → Press SPACE to capture (captures N images with delay)         ║
║     → Press Q when done with current subject                        ║
║     → Run again for next subject                                     ║
║                                                                      ║
║   OUTPUT:                                                            ║
║     my_dataset/                                                      ║
║       Ahmed/   → ahmed_001.png … ahmed_010.png                      ║
║       Omar/    → omar_001.png  … omar_010.png                       ║
║       ...                                                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import os
import sys
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────
PHONE_IP         = "192.168.1.7"      # ← Change to your phone's IP
PHONE_PORT       = 8080
IMAGES_PER_SUBJECT = 10               # captures per person
DATASET_DIR      = "my_dataset"       # output folder
CAPTURE_DELAY    = 1.5                # seconds between auto-captures
# ─────────────────────────────────────────────────────────────────

STREAM_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"


# ── Colors (BGR) ─────────────────────────────────────────────────
CLR_GREEN   = (50,  200,  80)
CLR_RED     = (50,   50, 220)
CLR_BLUE    = (220, 130,  50)
CLR_WHITE   = (255, 255, 255)
CLR_BLACK   = (  0,   0,   0)
CLR_YELLOW  = ( 30, 210, 240)
CLR_DARK    = ( 20,  20,  20)


def draw_ui(frame, subject_name, captured, total, status, guide_color, countdown=None):
    """
    Draws the full capture UI overlay on the frame.
    """
    h, w = frame.shape[:2]

    # ── Fingertip guide box ──────────────────────────────────────
    box_w, box_h = 280, 360
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    # Animated corner brackets
    L = 40  # corner length
    thickness = 3
    for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx*L, py), guide_color, thickness)
        cv2.line(frame, (px, py), (px, py + dy*L), guide_color, thickness)

    # Semi-transparent dark overlay outside the box
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)  # clear inside box
    frame = cv2.addWeighted(frame, 1.0, mask, 0.0, 0)
    # Darken outside
    outside = np.zeros_like(frame)
    cv2.rectangle(outside, (0,0), (w,h), (0,0,0), -1)
    cv2.rectangle(outside, (x1,y1), (x2,y2), (255,255,255), -1)
    outside_mask = outside[:,:,0] == 0
    frame[outside_mask] = (frame[outside_mask] * 0.45).astype(np.uint8)

    # ── Top banner ───────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w, 60), (15, 15, 15), -1)
    cv2.putText(frame, f"ENROLLING: {subject_name.upper()}",
                (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_GREEN, 2, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = w - 180, 15, 160, 28
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    filled = int(bar_w * (captured / total))
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), CLR_GREEN, -1)
    cv2.putText(frame, f"{captured}/{total}",
                (bar_x + bar_w//2 - 20, bar_y+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_WHITE, 1, cv2.LINE_AA)

    # ── Guide text above box ─────────────────────────────────────
    guide_text = "Place fingertip flat inside the box"
    tw = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    cv2.putText(frame, guide_text, (cx - tw//2, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 1, cv2.LINE_AA)

    # ── Countdown inside box ─────────────────────────────────────
    if countdown is not None:
        ctext = str(int(countdown) + 1)
        ts = cv2.getTextSize(ctext, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)[0]
        cv2.putText(frame, ctext,
                    (cx - ts[0]//2, cy + ts[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, guide_color, 6, cv2.LINE_AA)

    # ── Status bar at bottom ─────────────────────────────────────
    cv2.rectangle(frame, (0, h-55), (w, h), (15,15,15), -1)
    cv2.putText(frame, status, (15, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, "SPACE=Start Capture   Q=Done with this subject",
                (15, h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)

    return frame


def preprocess_and_save(frame, save_path):
    """
    Crop the center fingertip region, convert to grayscale,
    apply CLAHE to boost ridge contrast, save.
    Returns the processed image for preview.
    """
    h, w = frame.shape[:2]

    # Crop the guide box region
    box_w, box_h = 280, 360
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w, cx + box_w // 2)
    y2 = min(h, cy + box_h // 2)
    cropped = frame[y1:y2, x1:x2]

    # Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Resize to match SOCOFing-style dimensions
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # CLAHE — boosts ridge contrast dramatically
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Slight Gaussian blur to reduce noise (same as main.py preprocess)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    cv2.imwrite(save_path, enhanced)
    return enhanced


def show_preview(processed_img, subject_name, index):
    """Flash a quick preview of the captured & processed image."""
    preview = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    preview = cv2.resize(preview, (300, 300))
    label = f"Saved #{index} for {subject_name}"
    cv2.putText(preview, label, (10, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREEN, 1, cv2.LINE_AA)
    cv2.imshow("Last Capture", preview)
    cv2.waitKey(600)
    cv2.destroyWindow("Last Capture")


def collect_subject(cap, subject_name, subject_dir, images_per_subject):
    """
    Main collection loop for one subject.
    Returns number of images captured.
    """
    os.makedirs(subject_dir, exist_ok=True)
    captured = 0
    capturing = False
    capture_start_time = None

    print(f"\n[COLLECTING] Subject: {subject_name}")
    print(f"[INFO] Target: {images_per_subject} images → {subject_dir}")
    print(f"[INFO] Press SPACE to begin capturing, Q when done.\n")

    while captured < images_per_subject:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from phone camera.")
            time.sleep(0.1)
            continue

        now = time.time()

        if capturing:
            elapsed = now - capture_start_time
            remaining_captures = images_per_subject - captured
            which_in_burst = int(elapsed / CAPTURE_DELAY)

            # Time to capture?
            if which_in_burst > (captured - _burst_start) and captured < images_per_subject:
                idx = captured + 1
                filename = f"{subject_name.lower().replace(' ','_')}_{idx:03d}.png"
                save_path = os.path.join(subject_dir, filename)
                processed = preprocess_and_save(frame, save_path)
                captured += 1
                print(f"  [CAPTURED] {filename}")
                show_preview(processed, subject_name, idx)
                guide_color = CLR_GREEN

            # Countdown display
            time_to_next = CAPTURE_DELAY - (elapsed % CAPTURE_DELAY)
            countdown_val = time_to_next if captured < images_per_subject else None
            status = f"Auto-capturing... {remaining_captures} remaining  (stay still!)"
            guide_color = CLR_GREEN
            display = draw_ui(frame.copy(), subject_name, captured,
                              images_per_subject, status, guide_color,
                              countdown=time_to_next if captured < images_per_subject else None)
        else:
            status = "Ready — press SPACE to start capturing"
            guide_color = CLR_BLUE
            display = draw_ui(frame.copy(), subject_name, captured,
                              images_per_subject, status, guide_color)

        cv2.imshow(f"Fingerprint Collector — {subject_name}", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print(f"[INFO] Done with {subject_name}. Captured {captured} images.")
            break

        if key == ord(' ') and not capturing:
            capturing = True
            capture_start_time = time.time()
            _burst_start = 0  # will be used as closure
            print(f"[INFO] Capturing started for {subject_name}...")

    cv2.destroyAllWindows()
    return captured


# Hacky closure var — Python scoping fix
_burst_start = 0


def main():
    print("=" * 60)
    print("  FINGERPRINT DATASET COLLECTOR")
    print("  Biometrics Project — Custom Dataset Builder")
    print("=" * 60)
    print()
    print("  BEFORE YOU START:")
    print("  1. Install 'IP Webcam' from Play Store")
    print("  2. Open it → tap 'Start server'")
    print(f"  3. Edit PHONE_IP in this file to match your phone's IP")
    print(f"  4. Current PHONE_IP = {PHONE_IP}")
    print(f"     Stream URL = {STREAM_URL}")
    print()

    # ── Connect to phone camera ──────────────────────────────────
    print(f"[CONNECT] Connecting to phone at {STREAM_URL} ...")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print()
        print("[ERROR] Could not connect to phone camera!")
        print("  Checklist:")
        print("  • Is IP Webcam running and 'Start server' pressed?")
        print(f"  • Is PHONE_IP correct? (currently: {PHONE_IP})")
        print("  • Are phone and PC on the same Wi-Fi network?")
        print(f"  • Try opening {STREAM_URL} in your PC browser — you should see video")
        print()
        print("  Falling back to laptop webcam (index 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No camera available. Exiting.")
            sys.exit(1)
        print("[INFO] Using laptop webcam.")

    # Set highest resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("[OK] Camera connected!\n")

    # ── Subject collection loop ──────────────────────────────────
    os.makedirs(DATASET_DIR, exist_ok=True)
    total_subjects = 0

    while True:
        print("-" * 40)
        name = input("Enter subject name (or 'done' to finish): ").strip()
        if name.lower() == "done" or name == "":
            break

        subject_dir = os.path.join(DATASET_DIR, name)
        existing = len(os.listdir(subject_dir)) if os.path.exists(subject_dir) else 0
        if existing > 0:
            print(f"[WARN] {subject_dir} already has {existing} images.")
            choice = input("  Overwrite / add more? (y/n): ").strip().lower()
            if choice != 'y':
                continue

        n = collect_subject(cap, name, subject_dir, IMAGES_PER_SUBJECT)
        total_subjects += 1
        print(f"[DONE] {name}: {n} images saved to {subject_dir}/")

        another = input("\nAdd another subject? (y/n): ").strip().lower()
        if another != 'y':
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  DATASET COLLECTION COMPLETE")
    print("=" * 60)
    subjects = [d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))]
    total_images = 0
    for s in subjects:
        sdir = os.path.join(DATASET_DIR, s)
        n = len([f for f in os.listdir(sdir) if f.endswith('.png')])
        total_images += n
        print(f"  {s:<20} {n} images")
    print(f"  {'─'*35}")
    print(f"  Total: {len(subjects)} subjects, {total_images} images")
    print(f"  Saved to: {os.path.abspath(DATASET_DIR)}/")
    print()
    print("  Next step: run attendance_app_custom.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
