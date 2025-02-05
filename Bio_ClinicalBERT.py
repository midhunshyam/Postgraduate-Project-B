# Bio_ClinicalBERT.py

import os
import time
import pandas as pd
import torch
from torch.optim import AdamW, Adam, SGD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm 
import numpy as np

class BioClinicalBERTClassifier:
    def __init__(
        self, 
        model_name="emilyalsentzer/Bio_ClinicalBERT", 
        num_labels=2, 
        optimizer_class=AdamW, 
        optimizer_params={'lr': 1e-5},
        verbose=False  # Added verbose parameter
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.verbose = verbose  # Initialize verbose attribute
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        # Keep a copy of the initial state so we can re-init the model for CV.
        self._initial_state_dict = self.model.state_dict().copy()

        self.configure_optimizer()
        self.freeze_model_layers()
        self.unfreeze_classifier_layer()

    def configure_optimizer(self):
        # (Re)initialize the optimizer
        self.optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            **self.optimizer_params
        )

    def freeze_model_layers(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def unfreeze_classifier_layer(self, requires_grad=True):
        # Depending on the model architecture, 'classifier' may vary
        # For BertForSequenceClassification, it's 'classifier'
        if hasattr(self.model, 'classifier'):
            self.model.classifier.weight.requires_grad = requires_grad
            self.model.classifier.bias.requires_grad = requires_grad
        else:
            raise AttributeError("The model does not have a 'classifier' attribute.")

    def unfreeze_last_layers(self, n=1):
        # Unfreeze last n layers of the BERT encoder
        num_layers = len(self.model.bert.encoder.layer)
        if n > num_layers:
            raise ValueError(f"Cannot unfreeze {n} layers; the model only has {num_layers} layers.")
        for i in range(max(0, num_layers - n), num_layers):
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

    def check_layer_status(self):
        # Utility to check which layers are frozen/unfrozen
        for name, param in self.model.named_parameters():
            print(f"{name}: {'requires_grad = True' if param.requires_grad else 'requires_grad = False'}")

    def dataframe_to_dataloader(self, dataframe, batch_size=32, shuffle=True):
        if 'text' not in dataframe or 'label' not in dataframe:
            raise ValueError("Dataframe must contain 'text' and 'label' columns")

        # Make a copy to avoid SettingWithCopyWarning
        dataframe = dataframe.copy()

        # Convert labels to int if not integer dtype
        if not pd.api.types.is_integer_dtype(dataframe['label']):
            try:
                dataframe['label'] = dataframe['label'].astype(int)
            except ValueError:
                raise ValueError(
                    "Cannot convert 'label' column to int. "
                    "Ensure labels are valid integer-like values (e.g. 0, 1, 2)."
                )

        texts = dataframe['text'].tolist()
        labels = torch.tensor(dataframe['label'].tolist(), dtype=torch.int64)
        # One-hot encode labels
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()

        encoding = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )

        dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'], labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def _run_train_epoch(
        self, 
        data, 
        num_epochs=100, 
        batch_size=32, 
        test_split=None, 
        loss_threshold=None, 
        shuffle_train=True, 
        shuffle_test=False,
        cross_validation=False,
        cv_folds=5,
        verbose=False,
        print_every=10
    ):
        """
        Train the model using either a single train/test split or K-Fold cross-validation.
        
        Args:
            data (pd.DataFrame): Must contain columns ['text', 'label'].
            num_epochs (int): Number of epochs to train.
            batch_size (int): Batch size for dataloaders.
            test_split (float or None): Fraction for test split (ignored if cross_validation=True).
            loss_threshold (float or None): Early stopping threshold for train loss.
            shuffle_train (bool): Shuffle the training data in the dataloader.
            shuffle_test (bool): Shuffle the test data in the dataloader (if not cross-validation).
            cross_validation (bool): If True, use K-Fold cross-validation instead of a single split.
            cv_folds (int): Number of folds for K-Fold CV.
            verbose (bool): If True, prints training/test loss at specified intervals.
            print_every (int): Print losses every `print_every` epochs.
        """

        import time

        # Initialize lists to collect all true_labels and predictions
        all_true_labels = []
        all_predictions = []

        # If cross-validation is enabled, use KFold
        if cross_validation:
            all_fold_train_losses = []
            all_fold_test_losses = []
            total_train_time = 0.0
            total_test_time = 0.0

            # Prepare K-Fold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            data_indices = range(len(data))

            for fold, (train_idx, val_idx) in enumerate(kf.split(data_indices)):
                print(f"\n=== Starting fold {fold + 1}/{cv_folds} ===")

                # Re-initialize model weights from original state for each fold
                self.model.load_state_dict(self._initial_state_dict, strict=True)
                self.model.to(self.device)
                self.configure_optimizer()  # Re-initialize optimizer

                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]

                train_loader = self.dataframe_to_dataloader(train_data, batch_size=batch_size, shuffle=shuffle_train)
                val_loader = self.dataframe_to_dataloader(val_data, batch_size=batch_size, shuffle=shuffle_test)

                fold_train_losses = []
                fold_test_losses = []
                fold_train_time = 0.0
                fold_test_time = 0.0

                # ----- Training Loop -----
                for epoch in range(num_epochs):
                    self.model.train()
                    epoch_train_start = time.time()

                    train_epoch_loss = 0
                    for b_input_ids, b_input_mask, b_labels in train_loader:
                        b_input_ids = b_input_ids.to(self.device)
                        b_input_mask = b_input_mask.to(self.device)
                        b_labels = b_labels.to(self.device)

                        self.optimizer.zero_grad()
                        outputs = self.model(
                            b_input_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels
                        )
                        loss = outputs.loss
                        loss.backward()
                        self.optimizer.step()
                        train_epoch_loss += loss.item()

                    epoch_train_end = time.time()
                    fold_train_time += (epoch_train_end - epoch_train_start)

                    train_avg_loss = train_epoch_loss / len(train_loader)
                    fold_train_losses.append(train_avg_loss)

                    # Evaluate validation set every epoch
                    self.model.eval()
                    epoch_test_start = time.time()
                    test_epoch_loss, metrics = self.evaluate_loss(val_loader)
                    epoch_test_end = time.time()
                    fold_test_time += (epoch_test_end - epoch_test_start)

                    fold_test_losses.append(test_epoch_loss)

                    # Collect true labels and predictions
                    all_true_labels.extend(metrics['true_labels'])
                    all_predictions.extend(metrics['predictions'])

                    # Print training/test loss if requested and interval matches
                    if verbose and (epoch + 1) % print_every == 0:
                        print(
                            f"Fold {fold+1}/{cv_folds}, Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {train_avg_loss:.4f} | Test Loss: {test_epoch_loss:.4f}"
                        )

                    # Early stopping check
                    if loss_threshold and train_avg_loss < loss_threshold:
                        print(f"Early stopping triggered at epoch {epoch + 1} for fold {fold + 1}")
                        break

                # Store fold results
                all_fold_train_losses.append(fold_train_losses)
                all_fold_test_losses.append(fold_test_losses)
                total_train_time += fold_train_time
                total_test_time += fold_test_time

            # After all folds, compute average loss per epoch across folds
            max_epoch_len = max(len(f) for f in all_fold_train_losses)

            # Pad shorter folds with None so we can compute means safely
            padded_train = []
            padded_test = []
            for t_list, v_list in zip(all_fold_train_losses, all_fold_test_losses):
                if len(t_list) < max_epoch_len:
                    diff = max_epoch_len - len(t_list)
                    t_list += [None]*diff
                    v_list += [None]*diff
                padded_train.append(t_list)
                padded_test.append(v_list)

            # Convert to DataFrame to compute mean across folds
            df_train = pd.DataFrame(padded_train).T  # shape: epochs x folds
            df_test = pd.DataFrame(padded_test).T    # shape: epochs x folds

            avg_train_losses = df_train.mean(axis=1, skipna=True).tolist()
            avg_test_losses = df_test.mean(axis=1, skipna=True).tolist()

            # Plot average losses
            self.plot_loss_curves(avg_train_losses, avg_test_losses)

            print(f"Total train time (all folds): {total_train_time:.2f} sec")
            print(f"Total test time (all folds): {total_test_time:.2f} sec")

            # Return a dictionary with all necessary info
            return {
                'train_losses_avg': avg_train_losses,
                'test_losses_avg': avg_test_losses,
                'total_train_time': total_train_time,
                'total_test_time': total_test_time,
                'all_true_labels': all_true_labels,
                'all_predictions': all_predictions
            }

        else:
            # ---- Original Single Split Logic ----
            if isinstance(test_split, float):
                train_data, test_data = train_test_split(data, test_size=test_split, random_state=42)
                train_loader = self.dataframe_to_dataloader(train_data, batch_size=batch_size, shuffle=shuffle_train)
                test_loader = self.dataframe_to_dataloader(test_data, batch_size=batch_size, shuffle=shuffle_test)
            else:
                train_loader = self.dataframe_to_dataloader(data, batch_size=batch_size, shuffle=shuffle_train)
                test_loader = None

            self.model.train()
            train_loss_values = []
            test_loss_values = []

            total_train_time = 0.0
            total_test_time = 0.0

            all_true_labels = []
            all_predictions = []

            for epoch in range(num_epochs):
                # Training timing
                epoch_train_start = time.time()

                train_epoch_loss = 0
                for b_input_ids, b_input_mask, b_labels in train_loader:
                    b_input_ids = b_input_ids.to(self.device)
                    b_input_mask = b_input_mask.to(self.device)
                    b_labels = b_labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(
                        b_input_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    train_epoch_loss += loss.item()

                epoch_train_end = time.time()
                total_train_time += (epoch_train_end - epoch_train_start)

                train_avg_loss = train_epoch_loss / len(train_loader)
                train_loss_values.append(train_avg_loss)

                # Evaluate test data every 'print_every' epochs (if test_loader is available)
                if test_loader and (epoch + 1) % print_every == 0:
                    epoch_test_start = time.time()
                    test_epoch_loss, metrics = self.evaluate_loss(test_loader)
                    epoch_test_end = time.time()
                    total_test_time += (epoch_test_end - epoch_test_start)

                    test_loss_values.append(test_epoch_loss)

                    # Collect true labels and predictions
                    all_true_labels.extend(metrics['true_labels'])
                    all_predictions.extend(metrics['predictions'])
                else:
                    test_loss_values.append(None)  # for alignment

                # Print training loss and test loss if requested
                if verbose and (epoch + 1) % print_every == 0:
                    if test_loader:
                        last_test_loss = test_loss_values[-1]
                        if last_test_loss is not None:
                            print(
                                f"Epoch {epoch+1}/{num_epochs} | "
                                f"Train Loss: {train_avg_loss:.4f} | Test Loss: {last_test_loss:.4f}"
                            )
                        else:
                            print(
                                f"Epoch {epoch+1}/{num_epochs} | "
                                f"Train Loss: {train_avg_loss:.4f} | Test Loss: Not Evaluated This Epoch"
                            )
                    else:
                        print(
                            f"Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {train_avg_loss:.4f}"
                        )

                if loss_threshold and train_avg_loss < loss_threshold:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Plot
            non_null_test_losses = [x for x in test_loss_values if x is not None]
            self.plot_loss_curves(train_loss_values, non_null_test_losses)

            print(f"Total train time: {total_train_time:.2f} sec")
            print(f"Total test time: {total_test_time:.2f} sec")

            # Return a dictionary with all necessary info
            return {
                'train_losses': train_loss_values,
                'test_losses': test_loss_values,
                'total_train_time': total_train_time,
                'total_test_time': total_test_time,
                'all_true_labels': all_true_labels,
                'all_predictions': all_predictions
            }

    def evaluate_loss(self, test_loader):
        self.model.eval()
        total_loss = 0
        test_true_labels = []
        test_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = b_input_ids.to(self.device)
                b_input_mask = b_input_mask.to(self.device)
                b_labels = b_labels.to(self.device)

                outputs = self.model(
                    b_input_ids, 
                    attention_mask=b_input_mask, 
                    labels=b_labels
                )
                total_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                test_predictions.extend(preds.cpu().numpy())
                test_true_labels.extend(b_labels.argmax(dim=1).cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(test_true_labels, test_predictions)
        return avg_loss, {
            'true_labels': test_true_labels, 
            'predictions': test_predictions, 
            'accuracy': accuracy
        }

    def print_classification_report(self, true_labels, predictions, save_figure=False, figure_path='confusion_matrix_%j.png'):
        """
        Prints classification report and plots a confusion matrix.
        
        Args:
            true_labels (list or array): Ground truth labels.
            predictions (list or array): Predicted labels.
            save_figure (bool): If True, saves the confusion matrix plot to disk.
            figure_path (str): Path to save the confusion matrix plot. Use '%j' to include SLURM job ID.
        """
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=['Class 0', 'Class 1'], 
            output_dict=False
        )
        print("Classification Report:\n", report)
    
        # Plot confusion matrix using seaborn
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
    
        if save_figure:
            # Replace '%j' with SLURM_JOB_ID if present
            slurm_job_id = os.getenv("SLURM_JOB_ID", "noid")
            if "%j" in figure_path:
                figure_path = figure_path.replace("%j", slurm_job_id)
            plt.savefig(figure_path, bbox_inches='tight', dpi=300)
            print(f"Confusion matrix saved to {figure_path}")
    
        plt.show()

    def plot_loss_curves(self, train_losses, test_losses, 
                         save_figure=False, 
                         figure_path='loss_curves_%j.png'):
        """
        Plots the train and test loss curves. 
        Optionally saves the figure to disk if save_figure=True.

        Args:
            train_losses (list): List of training losses (or array).
            test_losses (list): List of test losses (or array, possibly partial).
            save_figure (bool): If True, calls plt.savefig(figure_path).
            figure_path (str): File path to save the plot if save_figure is True.
        """
        import os

        # Suppose we want to use the SLURM job ID (if available)
        slurm_job_id = os.getenv("SLURM_JOB_ID", "")
        if "%j" in figure_path:
            # Replace %j with the job ID (if environment variable is found)
            figure_path = figure_path.replace("%j", slurm_job_id if slurm_job_id else "noid")

        # If train_losses or test_losses are Pandas Series, convert to list
        if hasattr(train_losses, 'tolist'):
            train_losses = train_losses.tolist()
        if hasattr(test_losses, 'tolist'):
            test_losses = test_losses.tolist()

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', color='blue')

        # Plot test losses if they exist
        if test_losses:
            if len(test_losses) == len(train_losses):
                plt.plot(test_losses, label='Test Loss', color='red')
            else:
                # For partial logs, only plot the points that are not None
                valid_points = [
                    (idx, val) for idx, val in enumerate(test_losses) if val is not None
                ]
                if valid_points:
                    xs, ys = zip(*valid_points)
                    plt.plot(xs, ys, label='Test Loss', color='red')

        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if save_figure:
            plt.savefig(figure_path, bbox_inches='tight', dpi=300)
            print(f"Loss curves saved to {figure_path}")

        plt.show()

    def predict(self, texts, batch_size=32):
        """
        Predicts labels for the given texts in batches and measures prediction time.
    
        Args:
            texts (str or list or pd.Series): Input text(s) to classify.
            batch_size (int): Number of samples per batch.
    
        Returns:
            numpy.ndarray: Predicted labels.
        """
        import time  # Ensure time is imported
    
        # Convert input to list if it's a single string or pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError("Input texts should be a string, list of strings, or a pandas Series.")
    
        # Tokenize all texts
        encoding = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
    
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
    
        # Create TensorDataset and DataLoader for batching
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        all_predictions = []
    
        self.model.eval()  # Set model to evaluation mode
    
        start_time = time.time()  # Start timing
    
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", disable=not self.verbose):
                batch_input_ids, batch_attention_mask = [b.to(self.device) for b in batch]
    
                outputs = self.model(
                    input_ids=batch_input_ids, 
                    attention_mask=batch_attention_mask
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.cpu().numpy())
    
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
    
        if self.verbose:
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")
    
        return np.array(all_predictions)
   

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        # No 'weights_only' argument in PyTorch. Just load the state dict.
        saved_state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(saved_state_dict, strict=False)
        self.model.to(self.device)
        self.unfreeze_classifier_layer()
        print("Model weights loaded successfully.")
