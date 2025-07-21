# Postgraduate Project B: Natural Language Processing for Clinical Notes

**Western Sydney University**  
School of Computer, Data and Mathematical Sciences  

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

This project implements a two‑stage fine‑tuning pipeline for clinical note classification using Bio_ClinicalBERT and a downstream sequence classifier. I:

1. **Preprocess** the MIMIC‑III clinical notes.  
2. **Pre‑train** Bio_ClinicalBERT on a larger dataset (44 k samples).  
3. **Fine‑tune** a sequence classifier on preprocessed MIMIC‑III data.  
4. **Evaluate** on held‑out MIMIC‑III splits (70‑30 train‑test & validation).  
5. **Fine‑tune again** on a private gold‑standard dataset and evaluate final performance.


## Directory Structure

.
├── Data/
│   └── MIMIC3_Train.csv
├── Notebooks/
│   ├── 01_Preliminary_BioClinicalBERT_Modeling.ipynb
│   └── 02_Data_Preprocessing_MIMICIII.ipynb
├── Scripts/
│   └── Sequence_Classifier.py
├── Figures/
│   ├── MIMIC3_Training_Loss_Curve.png
│   ├── MIMIC3_Validation_Confusion_Matrix.png
│   ├── MIMIC3-Fine-tune_Test_Confusion_Matrix.png
│   ├── GoldStandard-Fine-tune_Training_Loss_Curve.png
│   └── GoldStandard-Fine-tune_Validation_Confusion_Matrix.png
├── Outputs/
│   ├── MIMIC3__Training-Validation_Results.txt
│   ├── MIMIC3-Fine-tune_GoldStandard-Test_Results.txt
│   └── GoldStandard-Training-Validation_Results.txt
├── Docs/
│   ├── MIMIC3_Data_Preprocessing_Report.pdf
│   └── Postgraduate-Project-B-Report.pdf
└── README.md



## Dataset

- **`Data/MIMIC3_Train.csv`**  
  Refined, preprocessed MIMIC‑III clinical-note dataset (after tokenization, filtering, and feature engineering). Used for sequence classifier fine‑tuning.

## Data Preprocessing

1. Open **`Notebooks/02_Data_Preprocessing_MIMICIII.ipynb`** for the full preprocessing pipeline:  
   - Loading raw MIMIC‑III records.  
   - Text cleaning, tokenization, stop‑word removal.  
   - Statistical analysis and data refinement.  
2. A PDF summary of the code and results is in **`Docs/MIMIC3_Data_Preprocessing_Report.pdf`**.


## Preliminary Bio_ClinicalBERT Modeling

- **Notebook:** `Notebooks/01_Preliminary_BioClinicalBERT_Modeling.ipynb`  
- **Purpose:** Pre‑train Bio_ClinicalBERT on an external 44 k‑sample train set to warm‑start downstream tasks.  
- **Data:** Separate `train.csv` on the `preliminarymodel` branch (44 k observations).


## Sequence Classifier Fine‑Tuning

- **Script:** `Scripts/Sequence_Classifier.py`  
- **Description:** Loads `MIMIC3_Train.csv`, fine‑tunes Bio_ClinicalBERT representations with a classification head.  
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

1. **Initial fine‑tuning** on MIMIC‑III (as above).  
2. **Secondary fine‑tuning** on proprietary gold‑standard data—results cannot expose raw data.  
3. **Performance** is summarized in the gold‑standard validation outputs and metrics above.  
4. **Privacy Note:** Gold‑standard dataset is private; only model outputs and metrics are included.


References

MIMIC‑III Clinical Database
Johnson AEW, Pollard TJ, Shen L, et al. MIMIC‑III, a freely accessible critical care database — PhysioNet. https://mimic.physionet.org/

Alsentzer, Emily et al., "Publicly Available Clinical BERT Embeddings”, In Proceedings of the 2nd Clinical Natural Language Processing Workshop, Association for Computational Linguistics, June 2019. https://arxiv.org/abs/1904.03323 
