# Data Directory

This directory contains the datasets used for the **4DViTADHD** project - Multimodal Intermediate Fusion for ADHD Diagnosis using 4D Vision Transformer.

## Directory Structure

### `FMRI_data/`
Contains functional Magnetic Resonance Imaging (fMRI) data for ADHD subjects and controls.
- **Purpose**: 4D spatio-temporal brain imaging data used for deep learning-based ADHD diagnosis
- **Format**: Volumetric brain scans with temporal dimensions
- **Data Type**: fMRI time series data capturing neural activity patterns

### `tabular/`
Contains tabular/structured data (clinical scores, demographic information, etc.)
- **Purpose**: Non-imaging features and clinical assessments
- **Format**: CSV or similar tabular formats
- **Data Type**: Demographics, clinical scores, behavioral assessments, and other structured features

## Usage

These datasets are used in conjunction with the 4D Vision Transformer model for multimodal intermediate fusion in ADHD diagnosis. The model combines:
- **4D fMRI data** (spatial + temporal dimensions) for brain imaging analysis
- **Tabular data** (clinical and demographic features) for comprehensive patient assessment

## Data Preparation

Before using the data:
1. Ensure proper preprocessing of fMRI volumes
2. Normalize clinical and demographic features
3. Split data into training, validation, and test sets as specified in the project notebooks
4. Follow data privacy and ethical guidelines for sensitive medical information

## Citation

If you use this dataset, please cite the 4DViTADHD project appropriately.

---

For more information about the project, see the main repository README.
