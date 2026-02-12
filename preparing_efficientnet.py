import os
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path

def prepare_odelia_data():
    """
    Convert ODELIA MRI data to 2D images for EfficientNet training
    """
    
    # Paths
    data_root = Path("/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/Odelia_Data")
    output_root = Path("/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/processed_data")
    
    # Label mapping
    label_map = {
        0: "no_lesion",
        1: "benign",
        2: "malignant"
    }
    
    # Create output directories for train/val splits
    for split in ["train", "val"]:
        for label_name in label_map.values():
            (output_root / split / label_name).mkdir(parents=True, exist_ok=True)
    
    # Find all annotation.csv files
    annotation_files = list(data_root.rglob("annotation.csv"))
    print(f"Found {len(annotation_files)} annotation files")
    
    all_samples = []
    
    for ann_file in annotation_files:
        print(f"\nProcessing: {ann_file}")
        df = pd.read_csv(ann_file)
        
        # Get institution name from path
        institution = ann_file.parent.parent.name
        
        for _, row in df.iterrows():
            uid = row['UID']
            lesion = row['Lesion']
            
            # Path to MRI data
            mri_folder = data_root / institution / "data_unilateral" / uid
            
            if not mri_folder.exists():
                print(f"  ⚠️  Folder not found: {uid}")
                continue
            
            # Look for Post_2.nii.gz (middle post-contrast phase)
            post2_path = mri_folder / "Post_2.nii.gz"
            
            if not post2_path.exists():
                print(f"  ⚠️  Post_2.nii.gz not found for {uid}")
                continue
            
            try:
                # Load NIfTI file
                nii_img = nib.load(str(post2_path))
                img_data = nii_img.get_fdata()
                
                # Get middle slice (along z-axis, typically axis 2)
                mid_slice_idx = img_data.shape[2] // 2
                slice_2d = img_data[:, :, mid_slice_idx]
                
                # Normalize to 0-255
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255
                slice_2d = slice_2d.astype(np.uint8)
                
                # Convert to RGB (EfficientNet expects 3 channels)
                img_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
                img_pil = Image.fromarray(img_rgb)
                
                # Save info
                all_samples.append({
                    'uid': uid,
                    'lesion': lesion,
                    'label_name': label_map[lesion],
                    'img': img_pil,
                    'institution': institution
                })
                
                print(f"  ✅ {uid}: Lesion={lesion} ({label_map[lesion]})")
                
            except Exception as e:
                print(f"  ❌ Error processing {uid}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Total samples processed: {len(all_samples)}")
    
    # Split into train (80%) and val (20%)
    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Save images
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        print(f"\nSaving {split_name} images...")
        for i, sample in enumerate(samples):
            label_folder = output_root / split_name / sample['label_name']
            filename = f"{sample['uid']}.png"
            sample['img'].save(label_folder / filename)
            
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{len(samples)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for split_name in ["train", "val"]:
        print(f"\n{split_name.upper()}:")
        for label_name in label_map.values():
            folder = output_root / split_name / label_name
            count = len(list(folder.glob("*.png")))
            print(f"  {label_name}: {count} images")
    
    print(f"\n✅ Data preparation complete!")
    print(f"Output directory: {output_root}")
    
    return output_root

if __name__ == "__main__":
    output_dir = prepare_odelia_data()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Your data is ready at: {output_dir}")
    print(f"2. Update your breast_cancer_classifier.py:")
    print(f"   data_dir = '{output_dir}'")
    print(f"3. Run training:")
    print(f"   python breast_cancer_classifier.py")