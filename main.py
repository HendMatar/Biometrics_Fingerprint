"""
Dataset: SOCOFing (Sokoto Coventry Fingerprint Dataset)
Two Methods:
  Method 1: ORB keypoint descriptors + BFMatcher
  Method 2: Gabor Filter Bank features + cosine similarity

Requirements:
  - Min 30 subjects (100 for better accuracy)
  - 80/20 train/test split
  - Preprocessing, Feature Extraction, Matching
  - Gen/Imp distributions, ROC curve
  - D-prime, EER, TMR@FMR=1% and 0.01%
  - Rank-1 Identification Rate (TPIR), FPIR, FNIR
"""

import os
import sys
import random
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION — adjust paths here
# ─────────────────────────────────────────────
DATASET_ROOT = ".\SOCOFing"          # folder containing Real and Altered
NUM_SUBJECTS = 100                   
IMAGES_PER_SUBJECT = 10             # SOCOFing has 10 real images per subject
TRAIN_RATIO = 0.8
OUTPUT_DIR = "./results"



# ══════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════

def load_dataset(root: str, num_subjects: int, imgs_per_subject: int):
    """"
    Pulls 1 Real image + N Altered images for specific finger.
    """
    root_path = Path(root)
    real_dir = root_path / "Real"
    altered_base = root_path / "Altered"
    
    if not real_dir.exists():
        print(f"[ERROR] Cannot find: {real_dir}")
        print("Make sure DATASET_ROOT points to the SOCOFing folder.")
        sys.exit(1)

    raw_dataset = defaultdict(list)

    # Load Real Images
    for f in real_dir.iterdir():
        if f.suffix.lower() in (".bmp", ".png", ".jpg", ".tif", ".tiff"):
            finger_id = f.stem # e.g., "100__M_Left_index_finger"
            raw_dataset[finger_id].append(str(f))

    # Load Altered Images
    for subfolder in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        alt_dir = altered_base / subfolder
        if alt_dir.exists():
            for f in alt_dir.iterdir():
                if f.suffix.lower() in (".bmp", ".png", ".jpg", ".tif", ".tiff"):
                    # Strip the alteration tags to match the Real finger_id
                    finger_id = f.stem.replace("_CR", "").replace("_Obl", "").replace("_Zcut", "")
                    raw_dataset[finger_id].append(str(f))

    # Filter for fingers that have enough images to split
    valid_fingers = {k: v for k, v in raw_dataset.items() if len(v) >= imgs_per_subject}
    
    if len(valid_fingers) < num_subjects:
         print(f"[WARN] Only found {len(valid_fingers)} fingers with enough images.")
         num_subjects = len(valid_fingers)

    selected_keys = list(valid_fingers.keys())[:num_subjects]

    dataset = {}
    for key in selected_keys:
         # Limit to requested images_per_subject
         dataset[key] = valid_fingers[key][:imgs_per_subject]

    print(f"[INFO] Loaded {len(dataset)} unique fingers (classes), "
          f"{sum(len(v) for v in dataset.values())} images total.")
    return dataset


def split_dataset(dataset: dict, train_ratio: float):
    train, test = {}, {}
    for sid, paths in dataset.items():
        n_train = max(1, int(len(paths) * train_ratio))
        train[sid] = paths[:n_train]
        test[sid]  = paths[n_train:] if len(paths) > n_train else [paths[-1]]
    return train, test


# ══════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════

def preprocess(img_path: str, size=(128, 128)) -> np.ndarray:
    """
    Load → grayscale → resize → CLAHE → Gaussian blur → normalize.
    CLAHE improves ridge contrast.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Slight Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ══════════════════════════════════════════════
# METHOD 1 — ORB Features
# ══════════════════════════════════════════════

class ORBMethod:
    """
    ORB (Oriented FAST and Rotated BRIEF) keypoint descriptors.
    Matching via BFMatcher with Hamming distance.
    Gallery = mean descriptor of training images per subject.
    """
    def __init__(self, n_features=500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.gallery = {}   # {subject_id: [descriptors]}

    def extract(self, img: np.ndarray):
        kp, des = self.orb.detectAndCompute(img, None)
        return des 

    def enroll(self, train_set: dict):
        #Build gallery from training images.
        for sid, paths in train_set.items():
            descs = []
            for p in paths:
                img = preprocess(p)
                des = self.extract(img)
                if des is not None:
                    descs.append(des)
            if descs:
                # Stack all descriptors for subject
                self.gallery[sid] = np.vstack(descs)

    def match_score(self, probe_des, gallery_des):
        #Return similarity score (higher = more similar).
        if probe_des is None or gallery_des is None:
            return 0.0
        matches = self.bf.knnMatch(probe_des, gallery_des, k=2)
        #Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        return len(good)

    def identify(self, probe_path: str):
        #Return ranked list of (subject_id, score).
        img = preprocess(probe_path)
        probe_des = self.extract(img)
        scores = {}
        for sid, gal_des in self.gallery.items():
            scores[sid] = self.match_score(probe_des, gal_des)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def verification_scores(self, test_set: dict):
        """
        Returns genuine_scores, impostor_scores for ROC/EER calculation.
        Genuine: probe vs same subject gallery.
        Impostor: probe vs different subject gallery.
        """
        genuine, impostor = [], []
        sids = list(self.gallery.keys())
        for sid, paths in test_set.items():
            if sid not in self.gallery:
                continue
            for p in paths:
                img = preprocess(p)
                probe_des = self.extract(img)
                # Genuine
                gen_score = self.match_score(probe_des, self.gallery[sid])
                genuine.append(gen_score)
                # Impostors (sample 5 random subjects)
                others = [s for s in sids if s != sid]
                for other_sid in random.sample(others, min(5, len(others))):
                    imp_score = self.match_score(probe_des, self.gallery[other_sid])
                    impostor.append(imp_score)
        return np.array(genuine, dtype=float), np.array(impostor, dtype=float)


# ══════════════════════════════════════════════
# METHOD 2 — Gabor Filter Bank
# ══════════════════════════════════════════════

class GaborMethod:
    """
    Gabor filter bank extracts ridge orientation/frequency features. 
    Gallery = mean feature vector per subject.
    Matching via cosine similarity.

    Why this beats ORB:
    - Gabor filters are specifically designed for oriented textures (ridges).
    - The feature vector captures global ridge patterns, not local keypoints
      (which can fail on low-quality).
    - refer to bonus task (we used both to differentiate)
    """
    def __init__(self, num_orientations=8, num_scales=4):
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernels = self._build_kernels()
        self.gallery = {}   # {subject_id: mean_feature_vector}

    def _build_kernels(self):
        kernels = []
        for scale in range(self.num_scales):
            freq = 0.05 + scale * 0.05  # spatial frequencies
            for i in range(self.num_orientations):
                theta = i * np.pi / self.num_orientations
                ksize = 31
                sigma = 3.0 + scale * 1.5
                k = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta,
                    lambd=1.0 / freq,
                    gamma=0.5, psi=0, ktype=cv2.CV_32F
                )
                kernels.append(k)
        return kernels   # 32

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Apply each Gabor kernel and compute energy statistics."""
        img_f = img.astype(np.float32) / 255.0
        features = []
        for k in self.kernels:
            filtered = cv2.filter2D(img_f, cv2.CV_32F, k)
            # 4×4 block-level mean energy (captures spatial distribution)
            h, w = filtered.shape
            bh, bw = h // 4, w // 4
            for bi in range(4):
                for bj in range(4):
                    block = filtered[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw]
                    features.append(np.mean(np.abs(block)))
                    features.append(np.std(block))
        return np.array(features, dtype=np.float32)

    def enroll(self, train_set: dict):
        #Build gallery: avg feature vector per subject.
        for sid, paths in train_set.items():
            vecs = []
            for p in paths:
                img = preprocess(p)
                vecs.append(self.extract(img))
            if vecs:
                mean_vec = np.mean(vecs, axis=0)
                self.gallery[sid] = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def identify(self, probe_path: str):
        #Return ranked list of ubject_id --> score
        img = preprocess(probe_path)
        probe_vec = self.extract(img)
        probe_vec = probe_vec / (np.linalg.norm(probe_vec) + 1e-8)
        scores = {sid: self.cosine_sim(probe_vec, gvec)
                  for sid, gvec in self.gallery.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def verification_scores(self, test_set: dict):
        genuine, impostor = [], []
        sids = list(self.gallery.keys())
        for sid, paths in test_set.items():
            if sid not in self.gallery:
                continue
            for p in paths:
                img = preprocess(p)
                probe_vec = self.extract(img)
                probe_vec /= (np.linalg.norm(probe_vec) + 1e-8)
                
                genuine.append(self.cosine_sim(probe_vec, self.gallery[sid]))
                
                others = [s for s in sids if s != sid] #AKA Imposters
                for other_sid in random.sample(others, min(5, len(others))):
                    impostor.append(self.cosine_sim(probe_vec, self.gallery[other_sid]))
        return np.array(genuine), np.array(impostor)


# ══════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════

def compute_rank1(method, test_set: dict):
    """
    Rank-1 Identification Rate = fraction of probes where correct subject is ranked #1 in the gallery.
    compute FPIR and FNIR at decision threshold (95th percentile of impostor).
    """
    correct = 0
    total   = 0
    for sid, paths in test_set.items():
        if sid not in method.gallery:
            continue
        for p in paths:
            ranked = method.identify(p)
            total += 1
            if ranked and ranked[0][0] == sid:
                correct += 1
    rank1 = correct / total if total > 0 else 0
    return rank1, correct, total


def compute_eer(genuine_scores, impostor_scores):
    
    #Equal Error Rate: threshold where FMR ≈ FNMR.
    
    labels = np.concatenate([np.ones(len(genuine_scores)),
                              np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_thresh = thresholds[eer_idx]
    auc_val = auc(fpr, tpr)
    return eer, eer_thresh, fpr, tpr, auc_val


def compute_dprime(genuine_scores, impostor_scores):
    """
    D-prime: separation between genuine and impostor score distributions.
    """
    mu_g, mu_i = np.mean(genuine_scores), np.mean(impostor_scores)
    sg, si = np.std(genuine_scores), np.std(impostor_scores)
    dp = abs(mu_g - mu_i) / (np.sqrt(0.5 * (sg**2 + si**2)) + 1e-9)
    return dp

from scipy.interpolate import interp1d
def tmr_at_fmr(genuine_scores, impostor_scores, target_fmr):
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    scores = np.concatenate([
        genuine_scores,
        impostor_scores
    ])

    fpr, tpr, _ = roc_curve(labels, scores)

    interp = interp1d(
        fpr,
        tpr,
        bounds_error=False,
        fill_value=(tpr[0], tpr[-1])
    )

    return float(interp(target_fmr))


def compute_fpir_fnir(method, test_set: dict, threshold_pct=95):

    genuine_accepted = 0
    genuine_total = 0
    impostor_accepted = 0
    impostor_total = 0

    all_imp_scores = []

    # ─────────────────────────────────────
    # Collect impostor scores
    # ─────────────────────────────────────
    for sid, paths in test_set.items():
        for p in paths:
            ranked = method.identify(p)
            if not ranked:
                continue

            top_sid, top_score = ranked[0]

            # only use top 1 impostor decision
            if top_sid != sid:
                all_imp_scores.append(top_score)

    # Threshold
    if len(all_imp_scores) == 0:
        threshold = 0.5
    else:
        threshold = np.percentile(all_imp_scores, threshold_pct)

    print("Threshold:", threshold)

    if len(all_imp_scores) > 0:
        print("Max impostor:", np.max(all_imp_scores))
        print("Min impostor:", np.min(all_imp_scores))

    # ─────────────────────────────────────
    # Compute FPIR / FNIR using TOP 1 ONLY (collected above)
    # ─────────────────────────────────────
    for sid, paths in test_set.items():
        for p in paths:
            ranked = method.identify(p)
            if not ranked:
                continue

            top_sid, top_score = ranked[0]

            # Genuine trial
            if top_sid == sid:
                genuine_total += 1
                if top_score >= threshold:
                    genuine_accepted += 1

            # Impostor trial
            else:
                impostor_total += 1
                if top_score >= threshold:
                    impostor_accepted += 1

    fnir = 1 - (genuine_accepted / genuine_total) if genuine_total > 0 else 1.0
    fpir = impostor_accepted / impostor_total if impostor_total > 0 else 0.0

    return fpir, fnir

# ══════════════════════════════════════════════
# PLOTTINGS
# ══════════════════════════════════════════════

def plot_gen_imp(gen1, imp1, gen2, imp2, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gen, imp, title in zip(
        axes,
        [gen1, gen2], [imp1, imp2],
        ["Method 1: ORB", "Method 2: Gabor"]
    ):
        ax.hist(imp, bins=50, alpha=0.6, color="red",   label="Impostor", density=True)
        ax.hist(gen, bins=50, alpha=0.6, color="green", label="Genuine",  density=True)
        ax.set_title(f"Gen/Imp Distribution — {title}")
        ax.set_xlabel("Match Score")
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gen_imp_distributions.png"), dpi=150)
    plt.close()
    print("[SAVED] gen_imp_distributions.png")


def plot_roc(fpr1, tpr1, auc1, fpr2, tpr2, auc2, out_dir):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr1, tpr1, "b-", lw=2, label=f"ORB   (AUC={auc1:.3f})")
    plt.plot(fpr2, tpr2, "r-", lw=2, label=f"Gabor (AUC={auc2:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("True Match Rate (TMR = 1-FRR)")
    plt.title("ROC Curve — Fingerprint Verification")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print("[SAVED] roc_curve.png")


def print_metrics_table(name, rank1, correct, total, eer, dp,
                        tmr_1pct, tmr_001pct, fpir, fnir):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Rank-1 Identification Rate : {rank1*100:.2f}%  ({correct}/{total})")
    print(f"  EER                        : {eer*100:.2f}%")
    print(f"  D-prime                    : {dp:.4f}")
    print(f"  TMR @ FMR=1%               : {tmr_1pct*100:.2f}%")
    print(f"  TMR @ FMR=0.01%            : {tmr_001pct*100:.2f}%")
    print(f"  FPIR                       : {fpir*100:.2f}%")
    print(f"  FNIR                       : {fnir*100:.2f}%")
    print(f"{'='*55}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load & Split
    print(f"\n[STEP 1] Loading dataset from: {DATASET_ROOT}")
    dataset = load_dataset(DATASET_ROOT, NUM_SUBJECTS, IMAGES_PER_SUBJECT)
    train_set, test_set = split_dataset(dataset, TRAIN_RATIO)

    n_train = sum(len(v) for v in train_set.values())
    n_test  = sum(len(v) for v in test_set.values())
    print(f"[INFO] Train: {n_train} images | Test: {n_test} images")
    print(f"[INFO] Subjects: {len(dataset)} | Train ratio: {TRAIN_RATIO*100:.0f}%")

    # Method 1: ORB 
    print("\n[STEP 2] Method 1 — ORB: Enrolling gallery...")
    orb_method = ORBMethod(n_features=500)
    orb_method.enroll(train_set)

    print("[STEP 2] Method 1 — ORB: Computing identification...")
    rank1_orb, correct_orb, total_orb = compute_rank1(orb_method, test_set)

    print("[STEP 2] Method 1 — ORB: Computing verification scores...")
    gen_orb, imp_orb = orb_method.verification_scores(test_set)
    eer_orb, _, fpr_orb, tpr_orb, auc_orb = compute_eer(gen_orb, imp_orb)
    dp_orb     = compute_dprime(gen_orb, imp_orb)
    tmr1_orb   = tmr_at_fmr(gen_orb, imp_orb, 0.01)
    tmr001_orb = tmr_at_fmr(gen_orb, imp_orb, 0.0001)
    fpir_orb, fnir_orb = compute_fpir_fnir(orb_method, test_set)

    # Method 2: Gabor
    print("\n[STEP 3] Method 2 — Gabor: Enrolling gallery...")
    gabor_method = GaborMethod(num_orientations=8, num_scales=4)
    gabor_method.enroll(train_set)

    print("[STEP 3] Method 2 — Gabor: Computing identification...")
    rank1_gab, correct_gab, total_gab = compute_rank1(gabor_method, test_set)

    print("[STEP 3] Method 2 — Gabor: Computing verification scores...")
    gen_gab, imp_gab = gabor_method.verification_scores(test_set)
    eer_gab, _, fpr_gab, tpr_gab, auc_gab = compute_eer(gen_gab, imp_gab)
    dp_gab     = compute_dprime(gen_gab, imp_gab)
    tmr1_gab   = tmr_at_fmr(gen_gab, imp_gab, 0.01)
    tmr001_gab = tmr_at_fmr(gen_gab, imp_gab, 0.0001)
    fpir_gab, fnir_gab = compute_fpir_fnir(gabor_method, test_set)

    # Results (in terminal)
    print_metrics_table("Method 1: ORB Keypoints",
        rank1_orb, correct_orb, total_orb,
        eer_orb, dp_orb, tmr1_orb, tmr001_orb, fpir_orb, fnir_orb)

    print_metrics_table("Method 2: Gabor Filter Bank (improved)",
        rank1_gab, correct_gab, total_gab,
        eer_gab, dp_gab, tmr1_gab, tmr001_gab, fpir_gab, fnir_gab)

    # Grade boundary
    print(f"\n{'─'*55}")
    rate = rank1_gab * 100
    if rate > 90:
        grade_pct = 100
    elif rate > 80:
        grade_pct = 90
    elif rate > 70:
        grade_pct = 80
    else:
        grade_pct = 70
    print(f"  Best Method (Gabor) Rank-1: {rate:.1f}%  →  Evaluation limit: {grade_pct}%")
    print(f"{'─'*55}")

    # Plots
    print("\n[STEP 4] Generating plots...")
    plot_gen_imp(gen_orb, imp_orb, gen_gab, imp_gab, OUTPUT_DIR)
    plot_roc(fpr_orb, tpr_orb, auc_orb, fpr_gab, tpr_gab, auc_gab, OUTPUT_DIR)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
