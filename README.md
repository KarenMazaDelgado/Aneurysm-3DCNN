# NeuroScan: AI-Powered Brain Aneurysm Detection System

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://neuroscan-frontend.vercel.app/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![License](https://img.shields.io/badge/license-Research-orange)](LICENSE)

**NeuroScan** is a full-stack medical AI application for detecting brain vessel abnormalities (aneurysms) from MRA scans. This system provides an AI-powered triage tool to assist radiologists in detecting potentially life-threatening brain aneurysms.

🌐 **[Live Demo](https://neuroscan-frontend.vercel.app/)**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Frontend](#-frontend)
- [Backend](#-backend)
- [Model Training](#-model-training--research)
- [Performance](#-current-performance)
- [Limitations](#-limitations--bias-analysis)
- [Future Work](#-future-improvements)
- [Authors](#-authors)
- [References](#-references)

---

## 🎯 Project Overview

### Problem Statement

**To what extent can AI detect abnormal vessel patterns in brain MRA scans compared to manual radiologist review?**

Stroke prevention and early detection of vascular abnormalities rely heavily on accurate interpretation of brain MRA scans. Radiologists review these images manually, a process that is:

- Time-consuming and vulnerable to fatigue
- Affected by high patient volumes and long shifts (errors increase by 226% at high volumes)
- Subject to human error, especially for subtle findings

### Clinical Context

- **6.5 million** people in the US have unruptured brain aneurysms
- **30,000** ruptures occur annually with **50% mortality rate**
- Sub-Saharan Africa has **<1 radiologist per 500,000 people**

### Our Solution

**NeuroScan addresses these challenges by:**

- ✅ Flagging potential aneurysms for clinical review (does not replace radiologist judgment)
- ✅ Reducing diagnostic burden by pre-screening scans during high-volume shifts
- ✅ Working with lower-resolution data (64×64×64 voxels) to support under-resourced healthcare settings
- ✅ Achieving **83.72% sensitivity** on validation set, approaching clinical MRA standards (~95%)

---

## 📁 Repository Structure
```
Neuroscan-Frontend/
├── frontend/              # Next.js web application
│   ├── app/              # React components and pages
│   ├── public/           # Static assets and test samples
│   └── package.json      # Frontend dependencies
│
├── backend/              # Gradio inference server
│   ├── app.py           # Main inference script
│   ├── model.pth        # Trained model weights (133MB)
│   └── requirements.txt # Backend dependencies
│
├── model/               # Training code and experiments
│   ├── Project.ipynb   # Main training notebook
│   ├── DATA SET/       # VesselMNIST3D dataset
│   ├── RESULTS/        # Model checkpoints
│   └── VISUALS/        # Performance charts
│
└── README.md           # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neuroscan-frontend.git
cd neuroscan-frontend
```

### 2. Start the Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend will run on `http://localhost:7860`

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend will run on `http://localhost:3000`

### 4. Test the System

1. Navigate to `http://localhost:3000`
2. Download test samples from the UI (20 jumbled scans included)
3. Upload a `.nii` or `.nii.gz` file
4. View predictions and 3D visualization with confidence heatmaps

---

## 🖥️ Frontend

### Features

- 🔍 **Single Scan Analysis** with interactive 3D visualization
- 📊 **Batch Processing** mode for multiple scans
- ⚖️ **Side-by-Side Comparison** mode
- 📥 **Download Test Samples** (20 jumbled aneurysm/normal scans)
- 🎨 **Real-Time Heatmap** visualization showing detection confidence

### Tech Stack

| Technology | Purpose |
|-----------|---------|
| Next.js 16 | React framework with TypeScript |
| React 19 | UI components |
| Tailwind CSS | Styling |
| NIfTI.js | 3D medical image rendering |
| WebGL | Hardware-accelerated graphics |
| @gradio/client | Backend API communication |

### Key Files
```
frontend/
├── app/
│   ├── page.tsx                    # Main application logic
│   ├── components/
│   │   └── NiftiViewer.tsx        # 3D visualization component
│   └── globals.css                # Global styles
├── next.config.ts                 # Backend proxy config
└── public/
    └── test_samples_20.zip        # Test dataset
```

### Installation
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:7860
```

---

## ⚙️ Backend

### Overview

Lightweight Python inference server that loads the trained 3D ResNet model and processes uploaded NIfTI files.

### Features

- 📁 Accepts `.nii` or `.nii.gz` files
- 🔬 Returns binary classification (Aneurysm vs Normal) with confidence scores
- 🗺️ Generates attention heatmaps for visualization
- ☁️ Hosted on Hugging Face Spaces

### Tech Stack

| Technology | Purpose |
|-----------|---------|
| Gradio 6.0.1 | Web interface framework |
| PyTorch | Deep learning inference |
| MONAI | Medical imaging toolkit |
| nibabel | NIfTI file processing |
| NumPy | Numerical operations |

### Model Details

- **Architecture:** 3D ResNet-18 (MONAI)
- **Input:** 64×64×64 voxel 3D volumes (automatically resized)
- **Output:** Binary classification + confidence scores
- **Model Size:** 133MB
- **Inference Time:** ~2-3 seconds per scan

### Installation
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### API Endpoints
```python
# Gradio Interface
predict(nifti_file) -> (prediction, confidence, heatmap)
```

---

## 🧠 Model Training & Research

### Key Results

| Metric | Value |
|--------|-------|
| **Recall (Sensitivity)** | 83.72% |
| **Specificity** | 87.3% |
| **Precision** | 53.73% |
| **Overall Accuracy** | 90.05% |
| **Aneurysms Detected** | 36/43 |
| **Missed Aneurysms** | 7 |
| **False Positives** | 31 |

### Training Methodology

#### 1. Data Preprocessing
```python
# Normalization
pixel_values = pixel_values / 255.0  # [0, 255] → [0, 1]

# Data Augmentation
transforms = Compose([
    RandRotate90(prob=0.5, spatial_axes=(0, 1, 2)),
    RandFlip(prob=0.5, spatial_axis=0),
    RandGaussianNoise(prob=0.2),
    RandAdjustContrast(prob=0.3)
])
```

#### 2. Model Architecture
```
Input: 64×64×64×1 grayscale volume
    ↓
3D Convolutional Layers + Batch Norm + ReLU
    ↓
Max Pooling
    ↓
ResNet Blocks (18 layers) with skip connections
    ↓
Global Average Pooling
    ↓
Fully Connected Layer
    ↓
Output: 2-class probability [Healthy, Aneurysm]
```

#### 3. Training Configuration

- **Loss Function:** Weighted Cross-Entropy (class weight: 7.5×)
- **Optimizer:** Adam (lr=1e-4)
- **Batch Size:** 16
- **Epochs:** 20
- **Early Stopping:** Patience 5
- **Hardware:** Google Colab T4 GPU

#### 4. Model Versions

We trained 5 different models with varying class weights:

| Model | Class Weight | Accuracy | Recall | Precision | Missed |
|-------|-------------|----------|--------|-----------|--------|
| V1 | 8.0× | 86.65% | 81.40% | 44.87% | 8 |
| V2 | 12.0× | 89.53% | 83.72% | 52.17% | 7 |
| V3 | 9.5× | 90.58% | 81.40% | 55.56% | 8 |
| V4 | 8.5× | 84.82% | 81.40% | 41.18% | 8 |
| **V5** | **7.5×** | **90.05%** | **83.72%** | **53.73%** | **7** ✓ |

**Why V5?**
- Tied for best recall (83.72%) - catches the most aneurysms
- Only 7 missed aneurysms (fewest false negatives)
- Strong precision (53.73%) - balanced false alarm rate
- High overall accuracy (90.05%)

### Dataset

#### VesselMNIST3D

- **Source:** [MedMNIST v2](https://medmnist.com/)
- **Original:** IntrA: 3D Intracranial Aneurysm Dataset (Xi Yang et al., CVPR 2020)
- **Total Samples:** 1,908
  - Training: 1,335
  - Validation: 191
  - Test: 382
- **Resolution:** 64×64×64 voxels
- **Class Distribution:** 
  - Healthy: 339 (88.7%)
  - Aneurysm: 43 (11.3%)
  - **Imbalance Ratio:** 8:1

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Severe Class Imbalance (8:1)** | Weighted sampling + class weights + threshold optimization |
| **Small Image Resolution** | Aggressive data augmentation + 3D architecture |
| **Limited Training Data** | Early stopping + validation monitoring + regularization |
| **Inconsistent Results** | Random seed setting + multiple training runs |
| **Metric Selection** | Prioritized recall over accuracy (patient safety) |

### Tech Stack

- Python 3.8+
- PyTorch 1.13+
- MONAI 1.3+
- NumPy, pandas, scikit-learn
- matplotlib, seaborn
- Google Colab (T4 GPU)
- Jupyter Notebook

### Training Code
```bash
cd model
# Open Project.ipynb in Jupyter or Google Colab
# Ensure GPU runtime is enabled
```

### Directory Structure
```
model/
├── Project.ipynb          # Main training notebook
├── DATA SET/
│   └── vesselmnist3d.npz # Original dataset
├── RESULTS/
│   ├── model_v1.pth      # Model checkpoints
│   ├── model_v2.pth
│   ├── model_v3.pth
│   ├── model_v4.pth
│   └── model_v5.pth      # Best model (used in deployment)
├── VISUALS/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── test_samples/         # 20 anonymized test scans
```

---

## 📊 Current Performance

### Comparison with Clinical Standards

| Metric | NeuroScan (V5) | Clinical MRA Studies |
|--------|----------------|---------------------|
| **Sensitivity (Recall)** | 83.72% | ~95% |
| **Specificity** | 87.3% | N/A |
| **Precision** | 53.73% | N/A |
| **Overall Accuracy** | 90.05% | N/A |
| **Resolution** | 64×64×64 voxels | 512×512×200+ voxels |
| **Missed Aneurysms** | 7 out of 43 (16.3%) | ~5% |
| **False Positives** | 31 out of 339 (9.1%) | N/A |

### Performance Analysis

✅ **Strengths:**
- Strong performance given resolution constraints (64×64×64 vs 512×512×200+)
- High specificity (87.3%) reduces false alarm burden
- Viable as screening/triage tool to assist radiologists
- Prioritizes patient safety by minimizing false negatives

⚠️ **Gap Analysis:**
- **11+ percentage point gap** compared to clinical MRA standards (~95% sensitivity)
- Resolution limitation is primary bottleneck
- Would benefit from higher-quality training data

### Clinical Use Case

**Intended Role:** First-pass screening tool
```
Patient MRA Scan
      ↓
NeuroScan Analysis
      ↓
   ┌──────┴──────┐
   │             │
High Risk    Low Risk
   │             │
Radiologist  Standard
Priority     Review
Review       Queue
```

**Benefits:**
- Reduces radiologist workload during high-volume shifts
- Flags obvious cases for immediate attention
- Provides second opinion for borderline cases
- Supports under-resourced healthcare settings

---

## 🔬 Limitations & Bias Analysis

### Identified Limitations

#### 1. Low Resolution (64×64×64 voxels)

**Impact:** Limits detection of subtle aneurysms

**Comparison:**
- NeuroScan: 64×64×64 voxels
- Clinical MRA: 512×512×200+ voxels
- **Information loss:** ~98% reduction in voxel count

#### 2. Unknown Demographics

**Missing Information:**
- Patient age
- Sex/gender
- Ethnicity
- Geographic origin
- Hospital/scanner type

**Impact:** Reduces generalizability across diverse populations

#### 3. Class Imbalance (8:1 ratio)

**Challenge:** Only 43 aneurysm samples in test set

**Solutions Applied:**
- Weighted random sampling
- Class weights (7.5×)
- Threshold optimization
- Aggressive data augmentation

**Remaining Risk:** May not capture rare or atypical presentations

#### 4. Single Imaging Modality

**Limitation:** Trained only on MRA scans

**Impact:** May not generalize to:
- CTA (Computed Tomography Angiography)
- DSA (Digital Subtraction Angiography)
- Different MRI field strengths (1.5T vs 3T)

#### 5. Binary Classification

**Limitation:** Treats all aneurysms equally

**Missing Capabilities:**
- Size estimation (small vs large)
- Risk stratification (low vs high risk)
- Aneurysm localization (which vessel)
- Multiple aneurysm detection

### Bias Mitigation Strategies

✅ **What We Did:**

1. **Transparent Evaluation**
   - Reported all metrics (not just accuracy)
   - Tested 5 model versions
   - Documented all limitations

2. **Patient Safety Priority**
   - Optimized for high recall (83.72%)
   - Minimized false negatives (only 7 missed)
   - Accepted higher false positive rate

3. **Documented Limitations**
   - Clear communication of resolution constraints
   - Explicit warnings about demographic bias
   - Honest performance gap analysis

4. **Assistive Tool Design**
   - Designed to support (not replace) radiologists
   - Provides confidence scores
   - Enables human override

5. **Systematic Approach**
   - Used established architectures (ResNet-18)
   - Applied best practices for imbalanced data
   - Validated across multiple runs

### Potential Sources of Bias

| Bias Type | Source | Mitigation |
|-----------|--------|------------|
| **Selection Bias** | Unknown patient demographics | Document limitation, recommend diverse validation |
| **Measurement Bias** | Low resolution (64×64×64) | Acknowledge constraint, plan higher-res version |
| **Algorithmic Bias** | Class imbalance (8:1) | Weighted sampling, class weights, threshold tuning |
| **Deployment Bias** | Single modality (MRA only) | Clearly scope intended use case |

---

## 🚧 Future Improvements

### Short-Term (3-6 months)

- [ ] Train on higher-resolution scans (128×128×128 voxels)
- [ ] Expand test dataset with more diverse samples
- [ ] Add aneurysm localization heatmaps
- [ ] Implement confidence calibration
- [ ] Create detailed error analysis

### Medium-Term (6-12 months)

- [ ] Collect dataset with diverse patient demographics
- [ ] Test on multiple imaging modalities (MRA, CTA, DSA)
- [ ] External validation across different hospitals
- [ ] Add risk stratification (size/severity classification)
- [ ] Develop explainability features (Grad-CAM, attention maps)

### Long-Term (12+ months)

- [ ] Train on full-resolution clinical scans (512×512×200+)
- [ ] Conduct prospective clinical trials
- [ ] Integration with hospital PACS systems
- [ ] Multi-class classification (different aneurysm types)
- [ ] Real-time inference optimization
- [ ] FDA approval pathway exploration

### Research Directions

1. **Architecture Improvements:**
   - Attention mechanisms
   - Multi-scale feature fusion
   - Ensemble methods
   - 3D U-Net for segmentation

2. **Data Enhancements:**
   - Synthetic data generation (GANs)
   - Semi-supervised learning
   - Transfer learning from related tasks
   - Active learning for hard cases

3. **Clinical Integration:**
   - DICOM support
   - HL7 FHIR compatibility
   - Batch processing pipeline
   - Quality control metrics

---

## 👥 Authors

This project was developed by:

- **Folabomi Longe** - [GitHub](https://github.com/folabomi) | [LinkedIn](https://linkedin.com/in/folabomi)
- **Oluwatodimu Adegoke** - [GitHub](https://github.com/todimu) | [LinkedIn](https://linkedin.com/in/todimu)
- **Ousman Bah** - [GitHub](https://github.com/ousman) | [LinkedIn](https://linkedin.com/in/ousman)
- **Karen Maza Delgado** - [GitHub](https://github.com/karen) | [LinkedIn](https://linkedin.com/in/karen)
- **Maria Garcia** - [GitHub](https://github.com/maria) | [LinkedIn](https://linkedin.com/in/maria)
- **Chimin Liu** - [GitHub](https://github.com/chimin) | [LinkedIn](https://linkedin.com/in/chimin)

*Completed as part of the **[AI4ALL Ignite](https://ai-4-all.org/)** accelerator program, investigating AI's capability to detect brain vessel abnormalities compared to radiologist review.*

---

## 📚 References

### Dataset & Model

1. Yang, X., et al. (2020). "IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_IntrA_3D_Intracranial_Aneurysm_Dataset_for_Deep_Learning_CVPR_2020_paper.pdf)

2. Yang, J., et al. (2023). "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." *Scientific Data*. [Link](https://medmnist.com/)

3. MONAI Consortium. (2020). "MONAI: Medical Open Network for AI." [Documentation](https://monai.io/)

### Clinical Context

4. Hanna, T. N., et al. (2018). "The Effects of Fatigue from Overnight Shifts on Radiology Search Patterns and Diagnostic Performance." *Radiology*, 287(1), 91-98. [Paper](https://pubs.rsna.org/doi/10.1148/radiol.2017170900)

5. Ivanovic, V., et al. (2024). "Increased Study Volumes Are Associated with Increased Error Rates in Neuroradiology." *American Journal of Neuroradiology*. [Paper](https://www.ajnr.org/)

6. Vlak, M. H., et al. (2011). "Prevalence of unruptured intracranial aneurysms, with emphasis on sex, age, comorbidity, country, and time period: a systematic review and meta-analysis." *The Lancet Neurology*, 10(7), 626-636.

### Related Work

7. Ueda, D., et al. (2019). "Deep Learning for MR Angiography: Automated Detection of Cerebral Aneurysms." *Radiology*, 290(1), 187-194.

8. Timmins, K. M., et al. (2021). "Comparing methods of detecting and segmenting unruptured intracranial aneurysms on TOF-MRAs: The ADAM challenge." *NeuroImage*, 238, 118216.

---

## ⚖️ License

This project is licensed under a research-only license. The VesselMNIST3D dataset is based on the IntrA dataset (Xi Yang et al., CVPR 2020) and is used under the terms specified by MedMNIST.

### Usage Restrictions

- ✅ Academic research
- ✅ Educational purposes
- ✅ Non-commercial experimentation
- ❌ Clinical deployment without validation
- ❌ Commercial use without permission

---

## ⚠️ Disclaimer

**IMPORTANT: This tool is for research and experimental purposes only.**

- All predictions must be verified by qualified medical professionals
- NeuroScan is designed to complement, not replace, clinical judgment
- Not FDA approved or clinically validated
- Not intended for diagnostic use
- Performance may vary across different populations and imaging protocols

**Medical professionals should:**
- Use as a screening/triage tool only
- Independently review all flagged cases
- Not rely solely on AI predictions for patient care decisions
- Report any observed performance issues or biases

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where we need help:**
- Higher-resolution training data
- Multi-center validation datasets
- Alternative architectures (ViT, U-Net)
- Deployment optimization
- Clinical workflow integration
- Documentation improvements


---

## 🙏 Acknowledgments

- **AI4ALL** for providing the Ignite accelerator program
- **MedMNIST** team for curating and sharing the VesselMNIST3D dataset
- **MONAI** community for medical imaging tools
- Our mentors and instructors for guidance throughout the project
- Radiologists who provided domain expertise and feedback

---
