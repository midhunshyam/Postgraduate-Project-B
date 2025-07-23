# Postgraduate Project B: Natural Language Processing for Clinical Notes

**Western Sydney University** 

School of Computer, Data and Mathematical Sciences


Date: November 2024

## Table of Contents

1. [Overview](#overview)  
2. [Directory Structure](#directory-structure)  
3. [Dataset](#dataset)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Preliminary Bio_ClinicalBERT Modeling](#preliminary-bio_clinicalbert-modeling)  
6. [Sequence Classifier Fine‑Tuning](#sequence-classifier-fine-tuning)  
7. [Model Evaluation](#model-evaluation)  
8. [Final Model Workflow](#final-model-workflow)  
9. [References](#references)  

## Overview

This project implements a two‑stage fine‑tuning pipeline for clinical note classification using Bio_ClinicalBERT and a downstream sequence classifier. We:

1. **Preprocess** the MIMIC‑III clinical notes.  
2. **Pre‑train** Bio_ClinicalBERT on a larger dataset (44 k samples).  
3. **Fine‑tune** a sequence classifier on preprocessed MIMIC‑III data.  
4. **Evaluate** on held‑out MIMIC‑III splits (70‑30 train‑validation).  
5. **Fine‑tune again** on a private gold‑standard dataset and evaluate final performance.


## Directory Structure

- **Data/**
  - `MIMIC3_Train.csv`
- **Notebooks/**
  - `Preliminary_BioClinicalBERT_Modeling.ipynb`
  - `Data_Preprocessing_MIMICIII.ipynb`
- **Scripts/**
  - `Sequence_Classifier.py`
- **Figures/**
  - `MIMIC3_Training_Loss_Curve.png`
  - `MIMIC3_Validation_Confusion_Matrix.png`
  - `MIMIC3-Fine-tune_GoldStandard-Test_Confusion_Matrix.png`
  - `GoldStandard-Fine-tune_Training_Loss_Curve.png`
  - `GoldStandard-Fine-tune_Validation_Confusion_Matrix.png`
- **Outputs/**
  - `MIMIC3_Training-Validation_Results.txt`
  - `MIMIC3-Fine-tune_GoldStandard-Test_Results.txt`
  - `GoldStandard-Training-Validation_Results.txt`
- **Docs/**
  - `MIMIC3_Data_Preprocessing_Report.pdf`
  - `Postgraduate-Project-B-Report.pdf`
- `README.md`




## Dataset

- **`Data/MIMIC3_Train.csv`**  
  Refined, preprocessed MIMIC‑III clinical-note dataset (after tokenization, filtering, and feature engineering). Used for sequence classifier fine‑tuning.

## Data Preprocessing

1. Open **`Notebooks/Data_Preprocessing_MIMICIII.ipynb`** for the full preprocessing pipeline:  
   - Loading raw MIMIC‑III records.  
   - Text cleaning, tokenization, stop‑word removal.  
   - Statistical analysis and data refinement.  
2. A PDF summary of the code and results is in **`Docs/MIMIC3_Data_Preprocessing_Report.pdf`**.


## Preliminary Bio_ClinicalBERT Modeling

- **Notebook:** `Notebooks/Preliminary_BioClinicalBERT_Modeling.ipynb`  
- **Purpose:** Pre‑train Bio_ClinicalBERT on an external 44 k‑sample train set to warm‑start downstream tasks.  
- **Data:** Separate `train.csv` on the `preliminarymodel` branch (44 k observations).


## Sequence Classifier Fine‑Tuning

- **Script:** `Scripts/Sequence_Classifier.py`  
- **Description:** This script loads `MIMIC3_Train.csv` and fine-tunes `Bio_ClinicalBERT` by adding a classification head for downstream clinical prediction tasks. The model is trained with the last encoder layer unfrozen, allowing the fine-tuning of both the classification head and the final transformer block to better adapt the model to the target clinical dataset.
- **Hyperparameters & arguments** are defined at the top of the script.


## Model Evaluation

### MIMIC‑III Split (70‑30 train‑test + validation)
- **Training Loss Curve:**  
  `Figures/MIMIC3_Training_Loss_Curve.png`  
- **Validation Confusion Matrix:**  
  `Figures/MIMIC3_Validation_Confusion_Matrix.png`  
- **Results (train & validation metrics):**  
  `Outputs/MIMIC3__Training-Validation_Results.txt`  
- **Test Confusion Matrix (held‑out 30%):**  
  `Figures/MIMIC3-Fine-tune_Test_Confusion_Matrix.png`  
- **Test Metrics:**  
  `Outputs/MIMIC3-Fine-tune_GoldStandard-Test_Results.txt`

### Gold‑Standard Fine‑Tuning & Evaluation
- **Fine‑Tuning Loss Curve:**  
  `Figures/GoldStandard-Fine-tune_Training_Loss_Curve.png`  
- **Validation Confusion Matrix:**  
  `Figures/GoldStandard-Fine-tune_Validation_Confusion_Matrix.png`  
- **Training & Validation Metrics:**  
  `Outputs/GoldStandard-Training-Validation_Results.txt`


## Final Model Workflow

1. **Initial fine‑tuning** Fine-tune Bio_ClinicalBERT on MIMIC3_Train.csv by training a classification head with the last encoder layer unfrozen. This step adapts Bio_ClinicalBERT representations to the specific clinical prediction task.
2. **Evaluation and Metrics**
Evaluate the fine-tuned model using held-out validation data. Track metrics such as accuracy, F1-score, precision, recall, and confusion matrix analysis to assess model performance.
3.	**Model Saving and Deployment**
Save the trained model checkpoint for future inference. The model can be deployed for clinical note classification, decision support systems, or further fine-tuning on related tasks.
5. **Secondary fine‑tuning** on proprietary gold‑standard data—results cannot expose raw data.
6. **Performance** is summarized in the gold‑standard validation outputs and metrics above.  
7. **Privacy Note:** Gold‑standard dataset is private; only model outputs and metrics are included.

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

pandas>=1.3.0  
torch>=1.12.0  
scikit-learn>=1.0.0  
matplotlib>=3.4.0  
seaborn>=0.11.0  
transformers>=4.30.0 


## References

MIMIC‑III Clinical Database
Johnson AEW, Pollard TJ, Shen L, et al. MIMIC‑III, a freely accessible critical care database — PhysioNet. https://mimic.physionet.org/

Alsentzer, Emily et al., "Publicly Available Clinical BERT Embeddings”, In Proceedings of the 2nd Clinical Natural Language Processing Workshop, Association for Computational Linguistics, June 2019. https://arxiv.org/abs/1904.03323 
