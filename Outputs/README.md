This directory stores **text-based results and metrics** generated during the training, validation, and test phases of the `Sequential_Classifier` model.

## Output Files

- **`MIMIC3__Training-Validation_Results.txt`**  
  Contains training and validation metrics on the **MIMIC-III split**, including:  
  - Training time  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score

- **`MIMIC3-Fine-tune_GoldStandard-Test_Results.txt`**  
  Evaluation metrics from testing the **MIMIC-III–trained classifier** on the **Gold-Standard Test Set (Private Hospital Data)**.  
  This assesses the model's generalization performance before domain-specific fine-tuning.

- **`GoldStandard-Training-Validation_Results.txt`**  
  Training and validation metrics for the **Gold-Standard fine-tuning phase**.  
  The classifier is initialized with MIMIC-III fine-tuned weights and further adapted to the **Private Hospital Annotated Dataset (Gold-Standard Set)**.
