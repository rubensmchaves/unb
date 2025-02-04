# Data load
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Preprocess
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# Evaluate
import evaluate

# Train
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Inference
from transformers import pipeline


ds_source = "data/Dmoz-Science.csv"


###################################################################################################
# Labels mapping
###################################################################################################

def get_label2id(labels):
    """
    Generate a dictionary mapping each label to a unique integer ID.

    Args:
        labels (list): List of label names.

    Returns:
        dict: Mapping of label names to integer IDs.
    """
    return {label: idx for idx, label in enumerate(labels)}


def get_id2label(labels):
    """
    Generate a dictionary mapping each integer ID to its corresponding label.

    Args:
        labels (list): List of label names.

    Returns:
        dict: Mapping of integer IDs to label names.
    """
    return {idx: label for idx, label in enumerate(labels)}

###################################################################################################
# Corpus and dataset inicialization
###################################################################################################


def load_and_split_dataset(csv_file):
    """
    Load a dataset from a CSV file into a Hugging Face dataset, convert labels to integers,
    and split it into training (70%), validation (10%), and test (20%) sets.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        DatasetDict: A Hugging Face dataset dictionary with train, validation, and test splits.
        dict: Mapping of label names to integer IDs.
        dict: Mapping of integer IDs to label names.
    """

    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(csv_file)
    df, _df = train_test_split(df, test_size=0.95, random_state=42, stratify=df["class"])

    # Get unique labels and create a mapping
    unique_labels = sorted(df["class"].unique())  # Ensure consistent ordering
    label2id = get_label2id(unique_labels)
    id2label = get_id2label(unique_labels)

    # Replace "class" column values with integer IDs
    df["label"] = df["class"].map(label2id)

    # Keep only "text" and "label" columns
    df = df[["text", "label"]]

    # Split dataset: 70% train, 30% temp (validation + test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

    # Further split temp dataset: 10% validation, 20% test
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df["label"])

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    return dataset_dict, label2id, id2label  # Return dataset and labels mapping


###################################################################################################
# Preprocess
###################################################################################################

def preprocess_function(examples):
    """
    Preprocessing function to tokenize text and truncate sequences to be no longer than 
    the model's maximum input length.

    Args:
        examples (dict): A dictionary containing a batch of text samples with the key "text".

    Returns:
        dict: A dictionary containing tokenized inputs, including input IDs and attention masks.
    """
    return tokenizer(examples["text"], truncation=True)


###################################################################################################
# Evaluation
###################################################################################################

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.

    This function calculates F1-score by comparing the predicted labels with the true labels.
    The predictions are first converted to class indices using argmax, assuming they are 
    probability distributions over classes.

    Args:
        eval_pred (tuple): A tuple containing:
            - predictions (numpy.ndarray): Model output logits or probabilities.
            - labels (numpy.ndarray): True labels.

    Returns:
        dict: A dictionary with the computed accuracy score.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    cm = confusion_matrix(labels, predictions)  # Confusion Matrix
    
    # Return the metrics
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }



if __name__ == "__main__":

    # Data Load ######################################################################
    ds_source = "data/NSF.csv"
    model_dir = "model_NSF"

    # Load dataset
    train_set, eval_set, test_set, unique_labels = load_and_split_dataset(ds_source)

    # Compute the labels amount
    qt_labels = len(unique_labels)

    # Convert to Hugging Face datasets
    data, label2id, id2label = load_and_split_dataset(ds_source)
    print("")
    print("Training size:", len(data["train"]))
    print("Validation size:", len(data["validation"]))
    print("Testing size:", len(data["test"]))
    print("")
    print(data["train"][0])


    # Preprocess ####################################################################
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Train #########################################################################
    model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=qt_labels, id2label=id2label, label2id=label2id
    )    

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        push_to_hub=False,  # Disable pushing to the hub
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Inference ####################################################################

    # Load the trained model and tokenizer from your local directory
    classifier = pipeline("text-classification", model=model_dir)

    # Make predictions
    predictions = []
    labels = []

    for example in data["test"]:
        text = example["text"]
        label = example["label"]
        labels.append(label)

        # Get model prediction
        pred = classifier(text)[0]  # We take the first prediction (since classifier returns a list of predictions)
        predictions.append(pred["label"])


    # Convert labels and predictions to numeric
    labels = np.array(labels)
    predictions = np.array([label2id[pred] for pred in predictions])

    # # Calculate F1-micro and F1-macro
    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-micro: {f1_micro:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")

    # Compute the confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    unique_labels = list(label2id.keys())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()