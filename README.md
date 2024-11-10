# Postgraduate Project B Natural Language Tasks
### Natural Language Processing for Clinical Notes

### Western Sydney University: School of Computer, Data and Mathematical Sciences


#### Dataset:
train.csv
Statistically evaluated dataset after refined text preprocessing for fine tuning seqClassifer.py

#### Preprocessing algorithm: mimiciii_datapreprocess.ipynb
MIMIC-III dataset preprocessing and statistical analysis and evaluation and further data refinement


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

finalmodel_goldstdtrainlosscurve.png
Final model trained on Gold Standard data

finalmodel_output.txt
Final model: seqClassfier wts fined tuned on gold std

finalmodel_testcm.png
Final model - Test output (80-20split) gold std


