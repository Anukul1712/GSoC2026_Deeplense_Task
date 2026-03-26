I tried 75% and 90% masking for my MAE Pre-training, and here are the results for the Classification Task (TaskIX.A).
| Mask Percentage | Macro | no_sub | cdm | axion |
|---|---|---|---|---|
| 75 | 0.9927 | 0.9984 | 0.9869 | 0.9929|
| 90 | 0.9963 | 0.9998 | 0.9940 | 0.9952|

# ROC-AUC Curve for Classification with 90 percent Masked Pre-Training.
![ROC-AUC Curve 90 MAE](ROC_AUC_90_TaskIX.png)

# Confusion Matrix for Classification with 90 percent Masked Pre-Training.
![Confusion Matrix 90 MAE](Confusion_90_TaskIX.png)

# MAE Reconstruction with Masked Percentage of 75 percent.
![MAE Reconstruction 75](MAE_Reconstruct_75_Ex1.png)
![MAE Reconstruction 75](MAE_Reconstruct_75_Ex2.png)
![MAE Reconstruction 75](MAE_Reconstruct_75_Ex3.png)

# MAE Reconstruction with Masked Percentage of 90 percent.
![MAE Reconstruction 90](MAE_Reconstruct_Ex1.png)
![MAE Reconstruction 90](MAE_Reconstruct_Ex2.png)
![MAE Reconstruction 90](MAE_Reconstruct_Ex3.png)

# SR Task(Task IX.B) Loss Curve
![SR Loss Curve](LossCurve_SR_TaskIXB.png)

# Super Resolution Output and Comparison
![SR ouput](SR_Output.png)
