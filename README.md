# Prostate Imaging Models

The models in this repository were built solely as test cases to validate an end-to-end workflow for AI/ML modeling with medical image data. No effort was made to optimize model performance or accuracy, and thus models are not meant for clinical or research decision-making.

These tests involve building models with 3D prostate MR images to design and validate workflows for customers building AI/ML models with medical imaging data.

## Models

**Detection model**
- Objective: Predict if an image contains tumor(s) that are either clinically significant or not, where clinical significance refers to having a Gleason score greater than or equal to 7, volume greater than or equal to 0.5cc, and/or extraprostatic extension.
- Architecture: MONAI ResNet10
- Metrics: AUC, sensitivity, specificity

**Segmentation model**
- Objective: Segment tumor(s) in an image based on masks.
- Architecture: MONAI U-Net
- Metrics: Dice score

## Data

**ProstateX**
- Geert Litjens, Oscar Debats, Jelle Barentsz, Nico Karssemeijer, and Henkjan Huisman. "ProstateX Challenge data", The Cancer Imaging Archive (2017). DOI: 10.7937/K9TCIA.2017.MURS5CL

**Annotations**

- R. Cuocolo, A. Stanzione, A. Castaldo, D.R. De Lucia, M. Imbriaco, Quality control and whole-gland, zonal and lesion annotations for the PROSTATEx challenge public dataset, Eur. J. Radiol. (2021) 109647. [https://doi.org/10.1016/j.ejrad.2021.109647](https://doi.org/10.1016/j.ejrad.2021.109647)
- Github link: [https://github.com/rcuocolo/PROSTATEx_masks](https://github.com/rcuocolo/PROSTATEx_masks)

## Usage

View package requirements and install with

```bash
conda env create -f environment.yml
conda activate prostatemr
```
