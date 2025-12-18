import numpy as np
import torch
from scipy.ndimage import zoom

class CropROI:
    """
    Crop image to center on lesion
    """
    def __init__(self, margin=(10, 10, 3)):
        self.margin = margin # (H, W, D)

    def __call__(self, sample):
        image = sample["image"][0]
        mask = sample["mask"][0]

        # Find non-zero mask voxels
        coords = np.where(mask > 0)
        if len(coords[0] == 0):
            sample["image"] = np.expand_dims(image, 0)
            sample["mask"] = np.expand_dims(mask, 0)
            return sample 
        
        zmin, ymin, xmin = np.min(coords, axis=1)
        zmax, ymax, xmax = np.max(coords, axis=1)

        # Apply margin
        zmin = max(zmin - self.margin[2], 0)
        ymin = max(ymin - self.margin[1], 0)
        xmin = max(xmin - self.margin[0], 0)

        zmax = min(zmax + self.margin[2], image.shape[2] - 1)
        ymax = min(ymax + self.margin[1], image.shape[1] - 1)
        xmax = min(xmax + self.margin[0], image.shape[0] - 1)

        # Crop
        image = image[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
        mask = mask[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

        sample["image"] = np.expand_dims(image, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        return sample


class ResampleResize:
    """
    Resample image and mask to target voxel spacing and resize to fixed shape
    """
    def __init__(self, target_spacing=(0.5, 0.5, 3.0), target_shape=(96, 96, 16)):
        self.target_spacing = target_spacing # in mm
        self.target_shape = target_shape # (H, W, D)

    def __call__(self, sample):
        image = sample["image"][0]
        mask = sample["mask"][0]
        spacing = sample["spacing"]

        # Compute zoom factors
        zoom_factors = [s/c for s, c in zip(spacing, self.target_spacing)]

        # Resample
        image = zoom(image, zoom_factors, order=1) # linear
        mask = zoom(mask, zoom_factors, order=0) # nearest

        # Resize to target shape
        current_shape = image.shape
        zoom_factors_shape = [t/c for t, c in zip(self.target_shape, current_shape)]
        image = zoom(image, zoom_factors_shape, order=1)
        mask = zoom(mask, zoom_factors_shape, order=0)

        # Add back channel dim
        sample["image"] = np.expand_dims(image, 0).astype(np.float32)
        sample["mask"] = np.expand_dims(mask, 0).astype(np.float32)
        
        return sample
    
class Normalize:
    """
    Z-score normalizle intensities within volume
    """
    def __call__(self, sample):
        img = sample["image"][0]
        mean = img.mean()
        std = img.std() if img.std() > 0 else 1.0
        img = (img - mean) / std
        sample["image"] = np.expand_dims(img, 0).astype(np.float32)
        return sample
    
class ToTensor:
    """
    Convert image and mask to torch tensor
    """
    def __call__(self, sample):
        sample["image"] = torch.from_numpy(sample["image"]).float()
        sample["mask"] = torch.from_numpy(sample["mask"]).float()
        sample["cls_label"] = torch.tensor(sample["cls_label"]).long()
        sample["gleason_label"] = torch.tensor(sample["gleason_label"]).long()
        return sample

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample