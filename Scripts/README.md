This directory contains the core training and evaluation script for fine‑tuning Bio_ClinicalBERT on MIMIC‑III data.

## Sequence_Classifier.py

**Purpose:**  
Fine‑tune a binary classification head (and the last encoder layer) on top of Bio_ClinicalBERT using your preprocessed MIMIC‑III dataset. The script:

1. Loads the `emilyalsentzer/Bio_ClinicalBERT` tokenizer and model.  
2. Freezes all parameters except the classifier head and the last BERT encoder layer.  
3. Reads `Data/MIMIC3_Train.csv`, splits into train (70%) / test (30%).  
4. Tokenizes texts & builds PyTorch `DataLoader`s.  
5. Runs a multi‑epoch training loop (default 10 epochs, batch size 32, LR=1e‑5).  
6. Saves the fine‑tuned model state dict at the end of training.  
7. Evaluates on the held‑out test set, computing accuracy, classification report, and confusion matrix.  
8. Plots and saves:  
   - Training loss curve (`training_loss_curve.png`)  
   - Confusion matrix (`confusion_matrix.png`)  
