I tested my pipeline with two different numbers of epochs, 30 and 50.
These are the different results for both of them.
Reconstructions were just slightly improved, but there was a significant improvement with 50 epochs; we can still work with 30 epochs.

# ROC-AUC Curve for 30 epochs
![ROC-AUC Curve for 30 epochs](ROC_AUC_PINN_30Epochs.png)

# Training Curves for 30 epochs
![Training Curves for 30 epochs](PINN_30Epochs_Training.png)

# Reconstructed Images for 30 epochs
![Reconstructed Images for 30 epochs](PINN_30Epochs_Reconstructed.png)

# ROC-AUC Curve for 50 epochs for test(10% holdout dataset from train folder)
![ROC-AUC Curve for 50 epochs for test](ROC_AUC_PINN_50Epochs_Val.png)

# ROC-AUC Curve for 50 epochs for given val folder
![ROC-AUC Curve for 50 epochs for val](ROC_AUC_PINN_50Epochs_Tests.png)

# Training Curves for 50 epochs
![Training Curves for 50 epochs](PINN_50Epochs_Training.png)

# Reconstructed Images for 50 epochs
![Reconstructed Images for 50 epochs](PINN_50Epochs_Reconstructed.png)
