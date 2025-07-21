# Load libraries
# This script is used to fine-tune Bio_ClinicalBERT sequential classifer head (layer) and layer 11 of Bio_ClinicalBERT on the mimiciii_datapreprocess.ipynb final output data with 25k observations.
import os
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW


# Load the model from hugging face

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", clean_up_tokenization_spaces = True)
# Load the model
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(torch.cuda.is_available())

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier layer
model.classifier.weight.requires_grad = True
model.classifier.bias.requires_grad = True

# Unfreeze the last layer of the encoder
num_layers = len(model.bert.encoder.layer)  # Get the number of layers in the encoder
for i in range(num_layers - 1, num_layers):  # Unfreeze the last layer
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = True
        
# Load the train dataset
train = pd.read_csv("train.csv").dropna()

# Ensure that labels are integers
train['label'] = train['label'].astype(int)


# Display dataset information
print(f"Dataset info: \n {train.info()}")
print(f"Dataset first 5 rows: \n {train.head(5)}")

print(train['label'].value_counts())


# Extract texts and labels from the dataset
texts = train['text'].tolist()  # Convert the 'text' column to a list
labels = train['label'].tolist()  # Convert the 'label' column to a list

# Tokenize the dataset with padding, truncation, and conversion to tensors
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Convert labels to tensor
labels = torch.tensor(labels)  # Convert list of labels to a tensor

# Create TensorDataset using tokenized inputs and labels
data = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

# Create DataLoader for the entire dataset
dataLoader = DataLoader(data, batch_size=16)

# Set the model to evaluation mode
model.eval()


# Print all parameters and their requires_grad status
print("\n".join([f"Parameter: {name} | Requires Grad: {param.requires_grad}" for name, param in model.named_parameters()]))


# Split the data into 70% for training and 30% for testing
xTrain, xTest, yTrain, yTest = train_test_split(inputs['input_ids'], labels, test_size=0.3, random_state=13)
trainMasks, testMasks = train_test_split(inputs['attention_mask'], test_size=0.3, random_state=13)

# Create DataLoader for the training and testing datasets
trainData = TensorDataset(xTrain, trainMasks, yTrain)
testData = TensorDataset(xTest, testMasks, yTest)
trainDataLoader = DataLoader(trainData, batch_size=32)
testDataLoader = DataLoader(testData, batch_size=32)

# Fine-tuning settings
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 10
loss_values = []

# Measure training time
start_training_time = time.time()

# Fine-tuning loop (on the 20% training set)
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in trainDataLoader:
        bInputIds, bInputMask, bLabels = [b.to(device) for b in batch]  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(bInputIds, attention_mask=bInputMask, labels=bLabels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Calculate average loss for this epoch
    avg_epoch_loss = epoch_loss / len(trainDataLoader)
    loss_values.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {avg_epoch_loss:.4f}")

# Save the fine-tuned model weights after training

# Use os.path.expanduser to expand the '~' to the full path

# Define the save path with an epoch placeholder
model_save_path = "/home/22058122/BioClinicalBERT/fine_tune/fine_tuned_clinical_bert_sequentialclassifier_{epoch}.pt"

# Save the model state dict to the specified path
torch.save(model.state_dict(), model_save_path.format(epoch=epoch))

# Print the path to which the model was saved
print(f"Model weights saved to {model_save_path.format(epoch=epoch)}")

end_training_time = time.time()
training_time = end_training_time - start_training_time
print(f"Training completed in {training_time:.2f} seconds.")

# Plot loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('/home/22058122/training_loss_curve.png')  # Specify the path and filename
plt.show()

# Measure testing time
start_testing_time = time.time()

# Set model to evaluation mode for testing
model.eval()

# Function to predict labels using the model
def predictLabels(dataLoader):
    predictions = []
    for batch in dataLoader:
        bInputIds, bInputMask, _ = [b.to(device) for b in batch]  # Move data to GPU
        with torch.no_grad():
            outputs = model(bInputIds, attention_mask=bInputMask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # Move back to CPU for evaluation
            predictions.extend(preds)
    return predictions

# Predict labels for the test set (80% data)
yPred = predictLabels(testDataLoader)

end_testing_time = time.time()
testing_time = end_testing_time - start_testing_time
print(f"Testing completed in {testing_time:.2f} seconds.")

# Calculate accuracy
accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {accuracy:.4f}")

# Calculate misclassification error rate
misclassificationErrorRate = 1 - accuracy
print(f"Misclassification Error Rate: {misclassificationErrorRate:.4f}")

# Generate a classification report (precision, recall, F1-score)
report = classification_report(yTest, yPred, target_names=['Class 0', 'Class 1'])
print("\nClassification Report:\n")
print(report)

# Generate confusion matrix
confMatrix = confusion_matrix(yTest, yPred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confMatrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('/home/22058122/confusion_matrix.png')  # Specify the path and filename
plt.show()
