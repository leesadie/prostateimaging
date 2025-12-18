# Prostate Imaging Models

Tests from building models with 3D prostate MR images to design and validate an end-to-end workflow for modeling with medical image data.

## Models

**Detection model**
- Objective: Predict if an image contains tumor(s) that are either clinically significant or not, where clinical significance refers to having a Gleason score greater than or equal to 7, volume greater than or equal to 0.5cc, and/or extraprostatic extension.
- Architecture: MONAI ResNet10
- Metrics: AUC, sensitivity, specificity

**Classification model**
- Objective: Predict the Gleason grade group (1-5) of an image, where the Gleason grade group is a simplified method to classify aggressiveness, and describes the histologic pattern of glands in the tumor. Higher grade groups indicate greater aggression.
- Architecture: 
- Metrics: AUC

**Segmentation model**
- Objective: Segment tumor(s) in an image based on masks.
- Architecture: U-Net
- Metrics: Dice score

## Data

**ProstateX**
- Geert Litjens, Oscar Debats, Jelle Barentsz, Nico Karssemeijer, and Henkjan Huisman. "ProstateX Challenge data", The Cancer Imaging Archive (2017). DOI: 10.7937/K9TCIA.2017.MURS5CL

**Annotations**

- R. Cuocolo, A. Stanzione, A. Castaldo, D.R. De Lucia, M. Imbriaco, Quality control and whole-gland, zonal and lesion annotations for the PROSTATEx challenge public dataset, Eur. J. Radiol. (2021) 109647. [https://doi.org/10.1016/j.ejrad.2021.109647](https://doi.org/10.1016/j.ejrad.2021.109647)
- Github link: [https://github.com/rcuocolo/PROSTATEx_masks](https://github.com/rcuocolo/PROSTATEx_masks)

## Usage

View package requirements and install with

```
pip install -r requirements.txt
```
