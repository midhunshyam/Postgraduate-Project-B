This directory includes all training and evaluation plots.

## MIMIC‑III Split (70% for training and 30% for validation)

- `MIMIC3_Training_Loss_Curve.png`  
  Training loss across epochs during fine-tuning of the `Sequential_Classifier` model on the 70% training split of the MIMIC-III dataset.

- `MIMIC3_Validation_Confusion_Matrix.png`  
  Confusion matrix illustrating the performance of the `Sequential_Classifier` on the 30% validation split of the MIMIC-III dataset.

## MIMIC-III Fine-Tuned Sequential_Classifier Evaluation on Gold-Standard Hospital Data
- `MIMIC3-Fine-tune_GoldStandard-Test_Confusion_Matrix.png`  
  Confusion matrix showing the performance of the Sequential_Classifier on the Gold-Standard Test Set immediately after fine-tuning on MIMIC-III data. This evaluation serves as a baseline before further domain-specific fine-tuning using the Private Hospital Annotated Dataset (Gold-Standard Set).

## Gold‑Standard Fine‑Tuning 

This phase involves further fine-tuning the model on a Private Hospital Annotated Dataset (Gold-Standard Set) to adapt it to domain-specific clinical language and annotation style, following the initial MIMIC-III fine-tuning.

- `GoldStandard-Fine-tune_Training_Loss_Curve.png`  
  Training loss curve showing how the model’s error decreases over epochs during fine-tuning on the Gold-Standard (Private Hospital) dataset.

- `GoldStandard-Fine-tune_Validation_Confusion_Matrix.png`  
  Confusion matrix evaluating model predictions on the Gold-Standard validation set, reflecting classification performance after domain adaptation.

###Summary of Purpose:

This step ensures the `Sequential_Classifier` moves from general clinical language (MIMIC-III) to the target institution’s data distribution, improving real-world applicability.

