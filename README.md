# Fingerprint Attendance System — Setup & Usage Guide
## Biometrics Project — Scenario Evaluation (30%)

---

## Files Overview

```
To run locally:

your_project/
│
├── main.py                    
├── collect_dataset.py         ← Step 1: Build your own fingerprint dataset
├── attendance_app_custom.py   ← Step 2: Live attendance system
└── my_dataset/                
    ├── Ahmed/
    │   ├── ahmed_001.png
    │   └── ...
    ├── Omar/
    └── ...
```

---

## Step 0 — Install IP Webcam (Android)

1. Open **Play Store** on your Android phone
2. Search for **"IP Webcam"** by *Pavel Khlebovich* (free, ~10MB)
3. Open the app
4. Scroll to the bottom → tap **"Start server"**
5. You'll see something like:

   ```
   http://192.168.1.5:8080
   ```

6. Open that URL in your **PC browser** — you should see live video from your phone

> **Important:** Phone and PC must be on the **same Wi-Fi network**

---

## Step 1 — Set Your Phone IP

Open both `collect_dataset.py` and `attendance_app_custom.py` and change this line near the top:

```python
PHONE_IP = "192.168.1.5"   # ← replace with YOUR phone's IP
```

---

## Step 2 — Collect Your Dataset

```bash
python collect_dataset.py
```

### What happens:
- Opens your phone camera stream
- Shows a **guide box** — place your fingertip flat inside it
- Enter each subject's name when prompted
- Press **SPACE** to start capturing (captures 10 images automatically with 1.5s delay)
- Press **Q** when done with that person
- Repeat for all teammates (minimum 5 subjects)

### Tips for good captures:
| Do | Don't |
|----|-------|
| Press finger **flat** against a surface | Hold finger in the air |
| Use **flashlight** for extra light | Capture in dim light |
| Keep finger **still** during capture | Move during countdown |
| Capture from **slightly different angles** | Same exact angle every time |
| **Wipe finger** before scanning | Scan a wet/sweaty finger |
| Get **close** — fill the guide box | Keep finger far from camera |

### Output:
```
my_dataset/
├── Ahmed/
│   ├── ahmed_001.png   ← grayscale, 128×128, CLAHE-enhanced
│   ├── ahmed_002.png
│   └── ...
├── Omar/
└── ...
```

---

## Step 3 — Run the Attendance System

```bash
python attendance_app_custom.py
```

### What happens:
1. Loads your dataset from `my_dataset/`
2. Splits 70% for **enrollment** (gallery), 30% for **testing**
3. Runs a **quick accuracy test** on held-out images — prints results
4. Connects to your phone camera
5. Opens the **live attendance window**

### Controls:
| Key | Action |
|-----|--------|
| `SPACE` | Capture current frame and identify |
| `L` | Print attendance log to console |
| `R` | Re-enroll (restart app) |
| `Q` | Quit |

### Output:
- **Green overlay** = ACCESS GRANTED (person recognized)
- **Red overlay** = ACCESS DENIED (not recognized)
- All events logged to `attendance_log.csv` with timestamp

---

## Tuning the Threshold

If you're getting too many false denials or false accepts, adjust:

```python
MATCH_THRESHOLD = 0.55   # in attendance_app_custom.py
```

| Threshold | Effect |
|-----------|--------|
| Higher (0.7+) | Stricter — fewer false accepts, more false denials |
| Lower (0.4–) | More lenient — fewer false denials, more false accepts |

The accuracy test printed at startup will help you calibrate this.

---

## What Satisfies Each Rubric Point

| Requirement | How it's met |
|-------------|-------------|
| ≥ 5 subjects | You + 4 teammates enrolled in `my_dataset/` |
| Acquisition from Camera | `collect_dataset.py` captures from phone via IP Webcam |
| Segmentation | `preprocess_phone_image()` crops fingertip region, applies CLAHE |
| Recognition | `CustomGaborMethod` (extends `GaborMethod` from `main.py`) |
| Real-life application | Attendance logging with GRANTED/DENIED decision |
| Complete | End-to-end: enroll → live scan → decision → CSV log |

---

## Troubleshooting

**"Could not connect to phone camera"**
- Make sure IP Webcam's "Start server" is running
- Check that PHONE_IP matches the IP shown in the app
- Try opening `http://YOUR_IP:8080/video` in PC browser
- Both devices must be on same Wi-Fi

**"No valid subjects found"**
- Run `collect_dataset.py` first
- Each subject needs at least 2 images

**Low accuracy**
- Capture more images per subject (change `IMAGES_PER_SUBJECT = 15` or more)
- Better lighting — use phone flashlight or a desk lamp
- Lower `MATCH_THRESHOLD` slightly
- Make sure finger fills the guide box during capture

**App crashes on import**
- Make sure `main.py` is in the **same folder** as these files
