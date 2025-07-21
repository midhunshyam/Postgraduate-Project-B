This directory holds Jupyter notebooks demonstrating data preprocessing and preliminary model experiments.

## Notebooks

### 1. `Data_Preprocessing_MIMICIII.ipynb`

### **Full MIMIC-III Data Preprocessing Pipeline**

1. **Loading Raw Records**  
   - Importing the original MIMIC-III clinical notes and associated data.

2. **Text Cleaning**  
   - Standardizing and cleaning clinical text to prepare for downstream processing.

3. **Defining REGEX Patterns**  
   - Custom regular expressions (REGEX) developed in collaboration with **Dr. Jim Basilakis** (Co-Supervisor, Medical Doctor/Domain Expert) to identify relevant clinical terms and phrases.

4. **LDA-REGEX Integration for ICD-9 Code Labeling**  
   - Combining **Latent Dirichlet Allocation (LDA)** topic modeling with `REGEX` to semi-automatically annotate the dataset.  
   - Observations are labeled based on **ICD-9 codes related to Vehicle-Related Trauma**.

5. **Statistical Analysis & Data Refinement**  
   - Performing statistical checks, distribution analysis, and refining labels to ensure data quality and consistency before model training.

   
### 2. `Preliminary_BioClinicalBERT_Modeling.ipynb`  
   In the preliminary modeling stage, Bio_ClinicalBERT is pre-trained on a 44k-sample external dataset to initialize the classifier head while keeping all encoder layers frozen.

