import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from torch.nn import BCEWithLogitsLoss

# Configuration
MODEL_CKPT = "distilbert-base-uncased"
OUTPUT_DIR = "./best_multilabel_model"
MAX_SAMPLES = 50000  # Take a manageable subset to keep training feasible

# Our 6 target labels
LABEL_NAMES = [
    "toxicity",
    "hate_speech",
    "threat",
    "insult",
    "obscene",
    "sexual_content"
]

# Mapping from Civil Comments dataset columns to our labels
CIVIL_COMMENTS_MAPPING = {
    "toxicity": "toxicity",
    "identity_attack": "hate_speech",
    "threat": "threat",
    "insult": "insult",
    "obscene": "obscene",
    "sexual_explicit": "sexual_content"
}

def main():
    print("1. Loading dataset...")
    # Load Civil Comments dataset
    ds = load_dataset("google/civil_comments", split="train")
    df = pd.DataFrame(ds)
    
    # Shuffle and subset
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
    
    print("2. Preprocessing labels...")
    # Apply mapping and binarize at >= 0.5 threshold
    for raw_col, new_label in CIVIL_COMMENTS_MAPPING.items():
        df[new_label] = (df[raw_col] >= 0.5).astype(float)
        
    # Drop rows where text is null (just in case)
    df = df.dropna(subset=['text'])
    
    # Create the label matrix
    labels = df[LABEL_NAMES].values
    texts = df['text'].tolist()
    
    print(f"Data shape: {df.shape}")
    
    # Compute pos_weight for BCEWithLogitsLoss
    pos_weights = []
    for i, label in enumerate(LABEL_NAMES):
        pos_count = df[label].sum()
        neg_count = len(df) - pos_count
        weight = neg_count / max(pos_count, 1) # avoid division by zero
        pos_weights.append(weight)
        print(f"  {label}: {pos_count} positive samples (weight: {weight:.2f})")
        
    # Convert to tensor
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)

    # Train/Val split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    print("3. Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    
    def tokenize(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)
        
    train_encodings = tokenize(X_train)
    val_encodings = tokenize(X_val)
    
    class ToxicityDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
            return item

        def __len__(self):
            return len(self.labels)
            
    train_dataset = ToxicityDataset(train_encodings, y_train)
    val_dataset = ToxicityDataset(val_encodings, y_val)
    
    print("4. Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(LABEL_NAMES),
        problem_type="multi_label_classification"
    )
    
    # Custom Trainer to inject pos_weight
    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Move pos_weight_tensor to the same device as logits
            loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
            
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Apply sigmoid to convert logits to probabilities
        probs = 1 / (1 + np.exp(-predictions))
        
        # Binarize for F1/Precision/Recall using 0.5 threshold
        preds = (probs >= 0.5).astype(int)
        
        metrics = {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)
        }
        
        for i, label in enumerate(LABEL_NAMES):
            try:
                metrics[f"{label}_roc_auc"] = roc_auc_score(labels[:, i], probs[:, i])
                metrics[f"{label}_pr_auc"] = average_precision_score(labels[:, i], probs[:, i])
                metrics[f"{label}_precision"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
                metrics[f"{label}_recall"] = recall_score(labels[:, i], preds[:, i], zero_division=0)
            except ValueError:
                # Can happen if a label has only 0s in the validation set
                pass
                
        return metrics

    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1"
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("5. Training...")
    trainer.train()

    print("6. Saving Model and Config...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label config
    label_config = {
        "labels": LABEL_NAMES,
        "thresholds": {label: 0.50 for label in LABEL_NAMES}
    }
    with open(os.path.join(OUTPUT_DIR, "label_config.json"), "w") as f:
        json.dump(label_config, f, indent=2)
        
    print(f"Model and label config saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
