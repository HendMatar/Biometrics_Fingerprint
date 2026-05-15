"""
Fingerprint Team Identification System
======================================
- Dataset: folder of colour phone images, one subfolder per teammate
- Enroll: builds a gallery from all images in each teammate's folder
- Match: given a probe image path, identifies the teammate (or rejects)

Usage:
  python fingerprint_matcher.py --dataset ./dataset --probe ./probe.jpg

"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MATCH_THRESHOLD = 0.55

ROTATIONS = [-10, -5, 0, 5, 10]


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess(img_path: str, size=(256, 256)):

    img = cv2.imread(str(img_path))

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        2
    )

    return 255 - thresh


# =========================================================
# ROTATION
# =========================================================

def rotate_image(img, angle):

    h, w = img.shape
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        matrix,
        (w, h)
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

class HybridExtractor:

    def __init__(self):

        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8
        )

        self.gabor_kernels = self.build_gabor()

    def build_gabor(self):

        kernels = []

        for theta in np.arange(0, np.pi, np.pi / 8):

            kernel = cv2.getGaborKernel(
                (21, 21),
                4.0,
                theta,
                10.0,
                0.5,
                0,
                ktype=cv2.CV_32F
            )

            kernels.append(kernel)

        return kernels

    def extract(self, img):

        keypoints, descriptors = self.orb.detectAndCompute(
            img,
            None
        )

        gabor_features = []

        img_f = img.astype(np.float32) / 255.0

        for k in self.gabor_kernels:

            filtered = cv2.filter2D(
                img_f,
                cv2.CV_32F,
                k
            )

            gabor_features.append(np.mean(filtered))
            gabor_features.append(np.std(filtered))

        gabor_vec = np.array(
            gabor_features,
            dtype=np.float32
        )

        norm = np.linalg.norm(gabor_vec)

        if norm > 0:
            gabor_vec /= norm

        return {
            "descriptors": descriptors,
            "gabor": gabor_vec
        }


# =========================================================
# IDENTIFIER + VERIFIER
# =========================================================

class FingerprintVerifier:

    def __init__(self):

        self.extractor = HybridExtractor()

        # stores all users
        self.gallery = {}

    # =====================================================
    # ENROLL ALL USERS
    # =====================================================
    def enroll(self, dataset_path):

        dataset_path = Path(dataset_path)

        folders = [
            d for d in dataset_path.iterdir()
            if d.is_dir()
        ]

        if not folders:
            print("[ERROR] No user folders found.")
            sys.exit(1)

        print(f"\n[ENROLL] Enrolling {len(folders)} teammates")
        print("-" * 50)

        for user_folder in folders:

            user_name = user_folder.name
            self.gallery[user_name] = []

            images = [
                f for f in user_folder.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTS
            ]

            for img_path in images:

                try:
                    img = preprocess(img_path)
                    feat = self.extractor.extract(img)

                    self.gallery[user_name].append(feat)

                except Exception as e:
                    print(f"[WARN] {img_path.name}: {e}")

            print(
                f"{user_name:15} "
                f"({len(self.gallery[user_name])} images)"
            )

        print("\n[ENROLL] Gallery ready")

    # =====================================================
    # SIMILARITY
    # =====================================================
    def similarity(self, f1, f2):

        desc1 = f1["descriptors"]
        desc2 = f2["descriptors"]

        orb_score = 0.0

        if desc1 is not None and desc2 is not None:

            bf = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=True
            )

            matches = bf.match(desc1, desc2)

            if matches:

                distances = [
                    m.distance for m in matches
                ]

                orb_score = 1.0 - (
                    np.mean(distances) / 100.0
                )

                orb_score = max(
                    0.0,
                    orb_score
                )

        gabor_score = float(
            np.dot(
                f1["gabor"],
                f2["gabor"]
            )
        )

        return (
            0.7 * orb_score
            + 0.3 * gabor_score
        )

    # =====================================================
    # IDENTIFICATION (find who this is)
    # =====================================================
    def identify(self, probe_path):

        img = preprocess(probe_path)

        best_user = None
        best_score = -1.0

        for angle in ROTATIONS:

            rotated = rotate_image(
                img,
                angle
            )

            probe_feat = self.extractor.extract(
                rotated
            )

            for user, templates in self.gallery.items():

                for temp in templates:

                    score = self.similarity(
                        probe_feat,
                        temp
                    )

                    if score > best_score:
                        best_score = score
                        best_user = user

        if best_score >= MATCH_THRESHOLD:
            return best_user, best_score

        return None, best_score

    # =====================================================
    # VERIFICATION (is this claimed user?)
    # =====================================================
    def verify(self, probe_path, claimed_user):

        if claimed_user not in self.gallery:
            return False, 0.0

        img = preprocess(probe_path)

        best_score = -1.0

        for angle in ROTATIONS:

            rotated = rotate_image(
                img,
                angle
            )

            probe_feat = self.extractor.extract(
                rotated
            )

            for temp in self.gallery[claimed_user]:

                score = self.similarity(
                    probe_feat,
                    temp
                )

                best_score = max(
                    best_score,
                    score
                )

        return (
            best_score >= MATCH_THRESHOLD,
            best_score
        )


# =========================================================
# RUN IDENTIFICATION
# =========================================================

def run(verifier, probe):

    probe = Path(probe)

    if not probe.exists():
        print("[ERROR] Probe not found")
        return

    print(f"\n[IDENTIFY] {probe.name}")
    print("-" * 40)

    user, score = verifier.identify(probe)

    print(f"Score     : {score:.4f}")
    print(f"Threshold : {MATCH_THRESHOLD}")
    print("-" * 40)

    if user:
        print(f"🥳 IDENTIFIED: {user}")
    else:
        print("😭 NOT ENROLLED")


# =========================================================
# INTERACTIVE
# =========================================================

def interactive(verifier):

    print("\nInteractive Mode (type exit)")

    while True:

        p = input("Image path: ").strip()

        if p.lower() == "exit":
            break

        if not os.path.exists(p):
            print("Not found")
            continue

        run(verifier, p)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True
    )

    parser.add_argument(
        "--probe"
    )

    parser.add_argument(
        "--interactive",
        action="store_true"
    )

    args = parser.parse_args()

    verifier = FingerprintVerifier()
    verifier.enroll(args.dataset)

    if args.probe:
        run(verifier, args.probe)

    if args.interactive or not args.probe:
        interactive(verifier)


if __name__ == "__main__":
    main()




"""Methodology:
Designed for low-quality phone fingerprint images.
Uses preprocessing (CLAHE, bilateral filter, adaptive thresholding)to improve contrast and reduce noise.
rotation augmentation to handle finger angle changes.
hybrid ORB + Gabor features to capture both local ridge details and overall fingerprint texture for more robust identification."""