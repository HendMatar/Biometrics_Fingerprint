"""
Fingerprint Team Identification System
=======================================
- Dataset: folder of colour phone images, one subfolder per teammate
- Enroll: builds a gallery from all images in each teammate's folder
- Match: given a probe image path, identifies the teammate (or rejects)

Usage:
  python fingerprint_matcher.py --dataset ./dataset --probe ./probe.jpg

Dataset folder structure:
  dataset/
    ahmed/
      ahmed_1.jpg
      ahmed_2.jpg
      ...
    sara/
      sara_1.jpg
      ...

Requirements:
  pip install opencv-python numpy scipy scikit-learn
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MATCH_THRESHOLD = 0.998   # cosine similarity threshold for accept/reject
                          # increase to be stricter, decrease to be more lenient


# ══════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════

def preprocess(img_path: str, size=(128, 128)) -> np.ndarray:
    """
    Load colour image → grayscale → resize → CLAHE → Gaussian blur.
    CLAHE boosts ridge contrast in colour phone photos.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ══════════════════════════════════════════════
# FEATURE EXTRACTION — Gabor Filter Bank
# ══════════════════════════════════════════════

class GaborExtractor:
    """
    Gabor filter bank: captures ridge orientation and frequency patterns.
    Better than ORB for colour phone photos because it captures
    global texture structure rather than relying on sharp keypoints.
    """
    def __init__(self, num_orientations=8, num_scales=4):
        self.kernels = self._build_kernels(num_orientations, num_scales)

    def _build_kernels(self, num_orientations, num_scales):
        kernels = []
        for scale in range(num_scales):
            freq = 0.05 + scale * 0.05
            for i in range(num_orientations):
                theta = i * np.pi / num_orientations
                sigma = 3.0 + scale * 1.5
                k = cv2.getGaborKernel(
                    (31, 31), sigma, theta,
                    lambd=1.0 / freq,
                    gamma=0.5, psi=0, ktype=cv2.CV_32F
                )
                kernels.append(k)
        return kernels  # 32 kernels

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Apply each kernel → block-level mean energy → feature vector."""
        img_f = img.astype(np.float32) / 255.0
        features = []
        for k in self.kernels:
            filtered = cv2.filter2D(img_f, cv2.CV_32F, k)
            h, w = filtered.shape
            bh, bw = h // 4, w // 4
            for bi in range(4):
                for bj in range(4):
                    block = filtered[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw]
                    features.append(np.mean(np.abs(block)))
                    features.append(np.std(block))
        vec = np.array(features, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)


# ══════════════════════════════════════════════
# GALLERY (ENROLLMENT)
# ══════════════════════════════════════════════

class FingerprintGallery:
    """
    Builds and stores a gallery (one mean feature vector per teammate).
    """
    def __init__(self):
        self.extractor = GaborExtractor()
        self.gallery = {}  # {name: mean_feature_vector}
        self.enrolled_counts = {}  # {name: number of images used}

    def enroll_from_folder(self, dataset_path: str):
        """
        Enroll all teammates from a dataset folder.
        Each subfolder name = teammate name.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"[ERROR] Dataset folder not found: {dataset_path}")
            sys.exit(1)

        subfolders = [d for d in dataset_path.iterdir() if d.is_dir()]
        if not subfolders:
            print(f"[ERROR] No subfolders found in {dataset_path}")
            print("  Expected structure: dataset/<teammate_name>/<images>")
            sys.exit(1)

        print(f"\n[ENROLL] Enrolling {len(subfolders)} teammates from: {dataset_path}")
        print("-" * 50)

        for folder in sorted(subfolders):
            name = folder.name
            images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]

            if not images:
                print(f"  [SKIP] {name}: no images found")
                continue

            vecs = []
            for img_path in images:
                try:
                    img = preprocess(img_path)
                    vec = self.extractor.extract(img)
                    vecs.append(vec)
                except Exception as e:
                    print(f"  [WARN] Could not process {img_path.name}: {e}")

            if not vecs:
                print(f"  [SKIP] {name}: all images failed to process")
                continue

            mean_vec = np.mean(vecs, axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-8)
            self.gallery[name] = mean_vec
            self.enrolled_counts[name] = len(vecs)
            print(f"  ✓  {name:<20} ({len(vecs)} image{'s' if len(vecs)>1 else ''})")

        print(f"\n[ENROLL] Gallery ready: {len(self.gallery)} teammates enrolled.")

    def identify(self, probe_path: str, top_n: int = 3):
        """
        Match a probe image against the gallery.
        Returns list of (name, cosine_similarity) sorted best-first.
        """
        img = preprocess(probe_path)
        probe_vec = self.extractor.extract(img)

        scores = {}
        for name, gal_vec in self.gallery.items():
            scores[name] = float(np.dot(probe_vec, gal_vec))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


# ══════════════════════════════════════════════
# MATCH & DISPLAY
# ══════════════════════════════════════════════

def run_match(gallery: FingerprintGallery, probe_path: str):
    """Run identification on a single probe image and print results."""

    probe_path = Path(probe_path)
    if not probe_path.exists():
        print(f"[ERROR] Probe image not found: {probe_path}")
        return

    print(f"\n[MATCH] Probe image: {probe_path.name}")
    print("-" * 50)

    results = gallery.identify(str(probe_path), top_n=len(gallery.gallery))

    if not results:
        print("[ERROR] No matches found.")
        return

    best_name, best_score = results[0]
    second_score = results[1][1] if len(results) > 1 else 0.0

    margin = best_score - second_score

    accepted = (
    best_score >= MATCH_THRESHOLD and
    margin >= 0.03 and
    best_score >= 0.99
   )

    print("\n  Top matches:")
    for i, (name, score) in enumerate(results[:5], 1):
        bar = "█" * int(score * 40)
        marker = " ◀ BEST" if i == 1 else ""
        print(f"  {i}. {name:<20} {score:.4f}  {bar}{marker}")

    print(f"\n{'─'*50}")

    if accepted:
        print(f"  ✅  IDENTIFIED AS: {best_name.upper()}")
        print(f"      Confidence : {best_score:.4f}")
        print(f"      Margin     : {margin:.4f}")
        print(f"      Threshold  : {MATCH_THRESHOLD}")
    else:
        print(f"  ❌  REJECTED — no confident match found")
        print(f"      Best score : {best_score:.4f}")
        print(f"      Second     : {second_score:.4f}")
        print(f"      Margin     : {margin:.4f}")
        print(f"      Threshold  : {MATCH_THRESHOLD}")
        print(f"      Closest    : {best_name}")

    print(f"{'─'*50}\n")

    return best_name if accepted else None

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    global MATCH_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Fingerprint Team Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll dataset and match one probe:
  python fingerprint_matcher.py --dataset ./dataset --probe ./test.jpg

  # Enroll dataset and enter interactive mode:
  python fingerprint_matcher.py --dataset ./dataset --interactive

  # Adjust acceptance threshold (default 0.72):
  python fingerprint_matcher.py --dataset ./dataset --probe ./test.jpg --threshold 0.75
        """
    )
    parser.add_argument("--dataset",     required=True,  help="Path to dataset folder (subfolders = teammates)")
    parser.add_argument("--probe",       default=None,   help="Path to probe image to match")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive matching mode")
    parser.add_argument("--threshold",   type=float, default=None,
                        help=f"Cosine similarity accept threshold (default: {MATCH_THRESHOLD})")
    args = parser.parse_args()

    
    if args.threshold is not None:
        MATCH_THRESHOLD = args.threshold

    # Enroll
    gallery = FingerprintGallery()
    gallery.enroll_from_folder(args.dataset)

    if not gallery.gallery:
        print("[ERROR] No teammates enrolled. Check your dataset folder.")
        sys.exit(1)

    # Match
    if args.probe:
        run_match(gallery, args.probe)

    if args.interactive or not args.probe:
        interactive_mode(gallery)


if __name__ == "__main__":
    main()