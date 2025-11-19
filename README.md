# Evaluating AI's Capability to Detect Brain Vessel Abnormalities Compared to Radiologist Review

*Investigated how well AI models can identify abnormal blood vessel patterns in brain MRA scans compared to traditional radiologist review, applying Python, deep learning methods, and biomedical imaging concepts within AI4ALL's cutting-edge AI4ALL Ignite accelerator.*

## Problem Statement <!--- do not change this line -->

*Stroke prevention and early detection of vascular abnormalities rely heavily on accurate interpretation of brain MRA scans. Radiologists review these images manually, a process that is time-consuming and vulnerable to fatigue or human error. With the increasing volume of patients and imaging data, there is a growing need for tools that can support radiologists and improve diagnostic consistency. This project explores whether AI — specifically deep learning models — can classify healthy and abnormal vessel segments with meaningful accuracy. Understanding AI's strengths and weaknesses in this task has real impact on clinical workflows, second-reader systems, and neurovascular diagnostics.*

## Key Results <!--- do not change this line -->

1. *Processed and explored the VesselMNIST3D dataset containing 3D brain vessel segments*
2. *Identified major sources of dataset imbalance, including a much larger number of healthy samples compared to aneurysm samples*
3. *Began prototyping a 3D-CNN (3D ResNet) to classify vessel patterns*
4. *Analyzed potential biases in the dataset:*
   - *Low image resolution limiting detection of subtle vessel abnormalities*
   - *Unknown demographics reducing generalizability across diverse populations*
   - *Class imbalance causing the model to favor predicting healthy vessels*
5. *Established baseline evaluation metrics (accuracy, precision, recall, FPR) to compare model performance with radiologist benchmarks*

## Methodologies <!--- do not change this line -->

*To accomplish this, we utilized Python to load, preprocess, and visualize 3D vessel segments from the VesselMNIST3D dataset. We built a 3D Convolutional Neural Network (3D-CNN), specifically a 3D ResNet architecture, to classify healthy vs abnormal vessel segments. The model was evaluated using metrics commonly used in medical imaging, including accuracy, recall, and false-positive rates. We identified dataset-level biases such as limited image resolution and imbalanced class distribution, and compared AI outputs to expected radiologist performance to understand the model's strengths and limitations.*

## Data Sources <!--- do not change this line -->

*- VesselMNIST3D Dataset: [MedMNIST](https://medmnist.com/)*
*- Intra: 3D Intracranial Aneurysm Dataset (Xi Yang et al., CVPR 2020)*
*- Related scientific literature provided through MedMNIST documentation and biomedical imaging research*

## Technologies Used <!--- do not change this line -->

- *Python*
- *PyTorch*
- *NumPy*
- *pandas*
- *3D-CNN / 3D ResNet*
- *Jupyter Notebook*
- *MedMNIST tools and utilities*

## Authors <!--- do not change this line -->

*This project was completed in collaboration with:*
- *Folabomi Longe*
- *Oluwatodimu Adegoke*
- *Ousman Bah*
- *Karen Maza Delgado*
- *Maria Garcia*
- *Chimin Liu*
