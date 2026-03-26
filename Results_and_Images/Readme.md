# Results and Images

This directory contains visual results, plots, and diagrams generated from the evaluation tasks.

## Folder Structure

```
Results_and_Images/
├── CommonTask_Results/      # Results for Task I: Multi-Class Classification
├── TaskVII_Results/         # Results for Task VII: Physics-Guided ML (PINN)
├── TaskIX_Results/          # Results for Task IX: Foundation Model (MAE & SR)
└── *.png                    # Pipeline diagrams and model architecture figures
```

## Description

- **CommonTask_Results/**: Contains performance metrics, ROC curves, and confusion matrices comparing the baseline models (ResNet-18, ResNet-34, EfficientNet-B3) on the common classification task.
- **TaskVII_Results/**: Contains visualizations for the Physics-Informed Neural Networks (PINN) approaches, including:
  - Training loss curves
  - ROC curves for Approach 1, 2, and 3
  - Physics visualizations: Potential maps ($\psi$), Deflection fields ($\alpha$), and Reconstruction residuals.
- **TaskIX_Results/**: Contains outputs from the Foundation Model experiments, including:
  - Masked Autoencoder (MAE) reconstruction examples.
  - Super-Resolution (SR) qualitative comparisons against bicubic baselines.
- **Root Images**: High-level pipeline diagrams referenced in the main project `README.md`.
