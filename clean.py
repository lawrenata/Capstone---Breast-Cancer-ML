import os
import re
import shutil
from pathlib import Path
import pandas as pd

# Best: stratified + grouped (balanced labels + no patient leakage)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False
    from sklearn.model_selection import GroupKFold

# -----------------
# CONFIG
# -----------------
MANIFEST_CSV = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/manifest.csv"
PNG_DIR      = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/all_axial_mips"
OUT_ROOT     = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/folds_with_test"

N_FOLDS      = 3          # CV folds on the 80% portion
TEST_FRAC    = 0.20       # fixed held-out test

RANDOM_STATE = 42

LESION_TO_CLASS = {
    0: "no_lesion",
    1: "benign",
    2: "malignant",
}
CLASSES = ["no_lesion", "benign", "malignant"]
# -----------------


def uid_from_png_name(png_name: str) -> str:
    """
    CAM_data_unilateral_ODELIA_BRAID1_0158_1_left_Post_1.png
    -> ODELIA_BRAID1_0158_1_left
    """
    stem = Path(png_name).stem  # remove .png

    m = re.search(r"(ODELIA_.*)$", stem)
    if not m:
        return ""
    stem = m.group(1)

    stem = re.sub(r"_(Pre|T2|Sub_1|Post_1|Post_2)$", "", stem)
    return stem


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def counts_uid(df: pd.DataFrame):
    return df["class_name"].value_counts().reindex(CLASSES, fill_value=0).to_dict()


def counts_imgs(folder: Path):
    out = {}
    total = 0
    for cls in CLASSES:
        n = len(list((folder / cls).glob("*.png")))
        out[cls] = n
        total += n
    return total, out


def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def main():
    manifest = pd.read_csv(MANIFEST_CSV)

    required = {"UID", "PatientID", "Lesion"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    manifest["Lesion"] = manifest["Lesion"].astype(int)
    manifest["class_name"] = manifest["Lesion"].map(LESION_TO_CLASS)
    if manifest["class_name"].isna().any():
        bad = manifest.loc[manifest["class_name"].isna(), "Lesion"].unique()
        raise ValueError(f"Unmapped Lesion codes found: {bad}. Fix LESION_TO_CLASS.")

    # Index PNGs by UID
    png_dir = Path(PNG_DIR)
    png_paths = list(png_dir.glob("*.png"))
    uid_to_pngs = {}
    for p in png_paths:
        uid = uid_from_png_name(p.name)
        if uid:
            uid_to_pngs.setdefault(uid, []).append(p)

    # Keep only UIDs that exist in both manifest and pngs
    manifest = manifest[manifest["UID"].isin(uid_to_pngs.keys())].copy()
    if manifest.empty:
        raise ValueError("No overlap between manifest UIDs and PNG filenames.")

    # Arrays for splitting
    X = manifest["UID"].to_numpy()
    y = manifest["Lesion"].to_numpy()
    groups = manifest["PatientID"].to_numpy()

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # 1) Make a fixed 20% TEST set (grouped)
    # ---------------------------------------
    # easiest robust way: use 5-way split -> 1 fold = 20%
    # (requires TEST_FRAC=0.20)
    if abs(TEST_FRAC - 0.20) > 1e-9:
        raise ValueError("This implementation assumes TEST_FRAC = 0.20. If you want a different fraction, say so.")

    if HAS_SGK:
        test_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        test_splits = list(test_splitter.split(X, y, groups=groups))
        print("Test split: StratifiedGroupKFold (balanced labels + no patient leakage).")
    else:
        test_splitter = GroupKFold(n_splits=5)
        test_splits = list(test_splitter.split(X, y, groups=groups))
        print("Test split: GroupKFold (no leakage; label balance may be worse).")

    # Pick the first fold as test (fixed via RANDOM_STATE when SGK available)
    train80_idx, test_idx = test_splits[0]

    train80 = manifest.iloc[train80_idx].copy()
    test_df = manifest.iloc[test_idx].copy()

    print("\nHeld-out TEST UID counts:", counts_uid(test_df))
    print("Remaining 80% UID counts:", counts_uid(train80))

    # Write TEST folder
    test_dir = out_root / "test"
    ensure_clean_dir(test_dir)
    for cls in CLASSES:
        (test_dir / cls).mkdir(parents=True, exist_ok=True)

    for _, r in test_df.iterrows():
        uid = r["UID"]
        cls = r["class_name"]
        for src in uid_to_pngs[uid]:
            dst = test_dir / cls / src.name
            copy_file(src, dst)

    t_total, t_cls = counts_imgs(test_dir)
    print("Held-out TEST images:", t_total, t_cls)

    # ---------------------------------------
    # 2) K-fold CV on the remaining 80%
    # ---------------------------------------
    X80 = train80["UID"].to_numpy()
    y80 = train80["Lesion"].to_numpy()
    g80 = train80["PatientID"].to_numpy()

    if HAS_SGK:
        cv_splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_splits = list(cv_splitter.split(X80, y80, groups=g80))
        print("\nCV split: StratifiedGroupKFold (balanced labels + no patient leakage).")
    else:
        cv_splitter = GroupKFold(n_splits=N_FOLDS)
        cv_splits = list(cv_splitter.split(X80, y80, groups=g80))
        print("\nCV split: GroupKFold (no leakage; label balance may be worse).")
        print("If you can: upgrade scikit-learn to get StratifiedGroupKFold.")

    # Build folds
    for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
        fold_dir = out_root / f"fold{fold_idx}"
        ensure_clean_dir(fold_dir)

        # Create base dirs
        for split_name in ["train", "val"]:
            for cls in CLASSES:
                (fold_dir / split_name / cls).mkdir(parents=True, exist_ok=True)

        train_rows = train80.iloc[tr_idx]
        val_rows = train80.iloc[va_idx]

        print(f"\nFold {fold_idx} UID counts:")
        print("  train:", counts_uid(train_rows))
        print("  val:  ", counts_uid(val_rows))

        # Copy images
        for split_name, rows in [("train", train_rows), ("val", val_rows)]:
            for _, r in rows.iterrows():
                uid = r["UID"]
                cls = r["class_name"]
                for src in uid_to_pngs[uid]:
                    dst = fold_dir / split_name / cls / src.name
                    copy_file(src, dst)
        # Image-level counts
        tr_total, tr_cls = counts_imgs(fold_dir / "train")
        va_total, va_cls = counts_imgs(fold_dir / "val")
        print("  images train:", tr_total, tr_cls)
        print("  images val:  ", va_total, va_cls)

    print(f"\nDone. Test + folds created under: {OUT_ROOT}")


if __name__ == "__main__":
    main()



# import os
# import re
# import shutil
# from pathlib import Path
# import pandas as pd


# # Try best splitter: stratified + grouped (prevents patient leakage AND balances labels)
# try:
#     from sklearn.model_selection import StratifiedGroupKFold
#     HAS_SGK = True
# except Exception:
#     HAS_SGK = False
#     from sklearn.model_selection import GroupKFold

# # -----------------
# # CONFIG (SET THESE)
# # -----------------
# MANIFEST_CSV = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/manifest.csv"          # has UID, PatientID, Lesion
# PNG_DIR      = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/all_axial_mips"        # all pngs here
# OUT_ROOT     = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/folds"  # will create fold0/1/2 here
# N_FOLDS      = 3

# # Confirm your lesion coding. Edit if needed.
# LESION_TO_CLASS = {
#     0: "no_lesion",
#     1: "benign",
#     2: "malignant",
# }
# # -----------------


# def uid_from_png_name(png_name: str) -> str:
#     """
#     CAM_data_unilateral_ODELIA_BRAID1_0158_1_left_Post_1.png
#     -> ODELIA_BRAID1_0158_1_left
#     """
#     stem = Path(png_name).stem  # remove .png

#     # keep substring starting at ODELIA_
#     m = re.search(r"(ODELIA_.*)$", stem)
#     if not m:
#         return ""  # no UID found
#     stem = m.group(1)

#     # remove known sequence suffixes at end
#     stem = re.sub(r"_(Pre|T2|Sub_1|Post_1|Post_2)$", "", stem)

#     return stem


# def copy_file(src: Path, dst: Path):
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if dst.exists():
#         return
#     shutil.copy2(src, dst)


# def main():
#     manifest = pd.read_csv(MANIFEST_CSV)

#     required = {"UID", "PatientID", "Lesion"}
#     missing = required - set(manifest.columns)
#     if missing:
#         raise ValueError(f"Manifest missing columns: {missing}")

#     manifest["Lesion"] = manifest["Lesion"].astype(int)
#     manifest["class_name"] = manifest["Lesion"].map(LESION_TO_CLASS)
#     if manifest["class_name"].isna().any():
#         bad = manifest.loc[manifest["class_name"].isna(), "Lesion"].unique()
#         raise ValueError(f"Unmapped Lesion codes found: {bad}. Fix LESION_TO_CLASS.")

#     # Index PNGs by UID
#     png_dir = Path(PNG_DIR)
#     png_paths = list(png_dir.glob("*.png"))
#     uid_to_pngs = {}
#     for p in png_paths:
#         uid = uid_from_png_name(p.name)
#         uid_to_pngs.setdefault(uid, []).append(p)

#     # Keep only UIDs that exist in both manifest and pngs
#     manifest = manifest[manifest["UID"].isin(uid_to_pngs.keys())].copy()
#     if manifest.empty:
#         raise ValueError("No overlap between manifest UIDs and PNG filenames.")

#     # Split inputs
#     X = manifest["UID"].to_numpy()
#     y = manifest["Lesion"].to_numpy()
#     groups = manifest["PatientID"].to_numpy()

#     if HAS_SGK:
#         splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
#         splits = list(splitter.split(X, y, groups=groups))
#         print("Using StratifiedGroupKFold (balanced labels + no patient leakage).")
#     else:
#         # Fallback: GroupKFold keeps patient together but may be less label-balanced.
#         splitter = GroupKFold(n_splits=N_FOLDS)
#         splits = list(splitter.split(X, y, groups=groups))
#         print("Using GroupKFold (no patient leakage; label balance may be worse).")
#         print("If you can: upgrade scikit-learn to get StratifiedGroupKFold.")

#     out_root = Path(OUT_ROOT)
#     out_root.mkdir(parents=True, exist_ok=True)

#     # Build folds
#     for fold_idx, (train_idx, val_idx) in enumerate(splits):
#         fold_dir = out_root / f"fold{fold_idx}"

#         # recreate fold directory
#         if fold_dir.exists():
#             shutil.rmtree(fold_dir)

#         # Create base dirs
#         for split_name in ["train", "val"]:
#             for cls in ["no_lesion", "benign", "malignant"]:
#                 (fold_dir / split_name / cls).mkdir(parents=True, exist_ok=True)

#         train_rows = manifest.iloc[train_idx]
#         val_rows = manifest.iloc[val_idx]

#         # sanity: print UID-level class counts
#         def counts(df):
#             return df["class_name"].value_counts().to_dict()

#         print(f"\nFold {fold_idx} UID counts:")
#         print("  train:", counts(train_rows))
#         print("  val:  ", counts(val_rows))

#         # Copy images
#         for split_name, rows in [("train", train_rows), ("val", val_rows)]:
#             for _, r in rows.iterrows():
#                 uid = r["UID"]
#                 cls = r["class_name"]

#                 for src in uid_to_pngs[uid]:
#                     dst = fold_dir / split_name / cls / src.name
#                     copy_file(src, dst)

#         # optional: report image-level counts too
#         def image_count(split_name):
#             total = 0
#             per_cls = {}
#             for cls in ["no_lesion", "benign", "malignant"]:
#                 n = len(list((fold_dir / split_name / cls).glob("*.png")))
#                 per_cls[cls] = n
#                 total += n
#             return total, per_cls

#         tr_total, tr_cls = image_count("train")
#         va_total, va_cls = image_count("val")
#         print("  images train:", tr_total, tr_cls)
#         print("  images val:  ", va_total, va_cls)

#     print(f"\nDone. Folds created under: {OUT_ROOT}")


# if __name__ == "__main__":
#     main()


# import os
# import numpy as np
# import nibabel as nib
# import imageio.v2 as imageio

# # ===== paths =====
# ROOT = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data"
# OUT_DIR = "/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data/all_axial_mips"
# os.makedirs(OUT_DIR, exist_ok=True)

# # ===== helper =====
# def save_axial_mip(nii_path, out_path):
#     img = nib.load(nii_path)
#     vol = img.get_fdata()  # shape: (X, Y, Z)

#     mip = np.max(vol, axis=2)  # axial MIP

#     # normalize to 0–255
#     mip -= mip.min()
#     if mip.max() > 0:
#         mip /= mip.max()
#     mip = (mip * 255).astype(np.uint8)

#     imageio.imwrite(out_path, mip)

# # ===== walk directory =====
# for root, _, files in os.walk(ROOT):
#     for f in files:
#         if f.endswith(".nii.gz"):
#             nii_path = os.path.join(root, f)

#             # unique filename
#             rel = os.path.relpath(nii_path, ROOT)
#             name = rel.replace("/", "_").replace(".nii.gz", ".png")

#             out_path = os.path.join(OUT_DIR, name)
#             save_axial_mip(nii_path, out_path)

# print("Done.")
