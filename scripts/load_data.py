from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import nibabel as nib
from collections import defaultdict

class MRIDataset(Dataset):
    """
    PyTorch dataset to load images, masks, and labels
    """
    def __init__(self, root_dir, labels_path, transform=None):
        self.root = root_dir
        self.img_dir = os.path.join(root_dir, "Images", "T2")
        self.mask_dir = os.path.join(root_dir, "Masks", "T2")
        self.transform = transform

        # Load labels
        df = pd.read_csv(labels_path)

        # Build dictionary for labels
        self.labels = {row["ID"]: row for _, row in df.iterrows()}

        # Get mask files
        mask_files = sorted([
            f for f in os.listdir(self.mask_dir) if f.endswith(".nii.gz")
        ])

        # Build dataset
        self.samples = []
        for mask_file in mask_files:
            # Extract finding_id from mask filename
            finding_id = mask_file.split("-t2")[0]

            # Replace the last hyphen with underscore to match CSV format
            parts = finding_id.rsplit("-", 1)
            if len(parts) == 2:
                finding_id = f"{parts[0]}_{parts[1]}"

            if finding_id not in self.labels:
                continue # Skip if no label

            # Extract case_id for matching image files
            case_id = finding_id.split("_Finding")[0]

            # Match image file with mask
            img_file = next(
                (f for f in os.listdir(self.img_dir) if f.startswith(case_id) and f.endswith(".nii.gz")),
                None
            )

            if img_file is None:
                continue

            self.samples.append({
                "finding_id": finding_id,
                "image_path": os.path.join(self.img_dir, img_file),
                "mask_path": os.path.join(self.mask_dir, mask_file)
            })

    def __len__(self):
        return len(self.samples)

    def load_nifti(self, path):
        nii = nib.load(path)
        arr = nii.get_fdata().astype(np.float32)
        return arr
    
    def __getitem__(self, index):
        sample = self.samples[index]
        finding_id = sample["finding_id"]
        label_row = self.labels[finding_id]

        # Load image and mask
        img = self.load_nifti(sample["image_path"])
        mask = self.load_nifti(sample["mask_path"])

        # Voxel spacing from header
        nii = nib.load(sample["image_path"])
        spacing = nii.header.get_zooms()[:3]

        # Add channel dim
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Labels
        clin_sig = label_row["Clinically Significant"]

        if isinstance(clin_sig, str):
            clin_sig = clin_sig.strip().lower() == "true"
        
        clin_sig = int(clin_sig) # 0 or 1

        gleason = label_row["Gleason Grade Group"]
        if isinstance(gleason, str) and "No biopsy" in gleason:
            gleason = -1
        else:
            gleason = int(gleason)

        sample_dict = {
            "image": img,
            "mask": mask,
            "cls_label": clin_sig,
            "gleason_label": gleason,
            "spacing": spacing,
            "id": finding_id
        }

        # Transform
        if self.transform:
            sample_dict = self.transform(sample_dict)

        return sample_dict
    

