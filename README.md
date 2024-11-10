# Postgraduate Project B Natural Language Tasks
### Natural Language Processing for Clinical Notes

### Western Sydney University: School of Computer, Data and Mathematical Sciences


#### Dataset:
train.csv
Statistically evaluated dataset after refined text preprocessing for fine tuning seqClassifer.py

#### Preprocessing algorithm: preprocess_mimiciii.ipynb
MIMIC-III dataset preprocessing and statistical analysis and evaluation and further data refinement
mimiciii_datapreprocess.pdf contains the codes for MIMIC III Clinical Database data preprocessing (ipynb takes long time to load)

#### Bio_ClinicalBERT - Preliminary modelling - premodel.ipynb
Uses different train.csv located in priliminary modelling git branch with 44k observations; https://github.com/midhunshyam/PPB/blob/preliminarymodel/train.csv)


#### Finetuning algorithm - Sequential Classifier (seqClassifier)

seqClassifier.py
Model finetuning algorithm

seqClassifier_losscurve.png
Sequential classifier (finetuned model) training loss curve

seqClassifier_output.txt
Output from running seqClassifier.py

seqClassifier_testconfusionmatrix.png
seqClassifier test confusion matrix (70-30split)


#### Finetuned model testing (seqClassifier on gold standard)

goldstdtest_finetuned.png
Goldstd test CM final tuned (seqClassifier.py)

goldstdtest_output.txt
Fine tuned model test output on gold standard data


#### Final model 

The pretrained Bio_ClinicalBERT undergoes fine-tuning twice - on MIMICIII extracted text and the goldstandard dataset to build the final model.
The goldstandard dataset is not available as it is subject to privacy protection. Only outputs without any details about the gold standard dataset is available. The final fine-tuning data is domain specific. The gold standard which is used in this project is specific for the industry and the specific client, and cannot be generalised. Alteration to the algorithm and data mining techniques should be expected based on domain.

finalmodel_goldstdtrainlosscurve.png
Final model trained on Gold Standard data

finalmodel_output.txt
Final model: seqClassfier wts fined tuned on gold std

finalmodel_testcm.png
Final model - Test output (80-20split) gold std


