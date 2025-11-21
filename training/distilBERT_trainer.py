# initial implementation of DistilBERT model

# using the Stanford IMDB dataset.
# Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd

import torch
import time

# helper functions

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

def add_label(dataset, split):
    label = 1 if "pos" in split else 0
    ds = dataset[split]
    ds = ds.add_column("label", [label] * len(ds))
    return ds

#get dataset from csv file
def load_soft_label_dataset(file_path):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda x: {
        "soft_label": [
            x["positive_prob"],
            x["negative_prob"],
            x.get("neutral_prob", 0.0)
            ]
        })
    return ds

#get IMDB dataset, create train and test datasets.
# this Was used in initial testing, should not be needed in final version
def load_imdb_dataset():
    dataset = load_dataset("text",
        data_files={
        "train_pos": "data/Imdb_v1/train/pos/*.txt",
        "train_neg": "data/Imdb_v1/train/neg/*.txt",
        "test_pos": "data/Imdb_v1/test/pos/*.txt",
        "test_neg": "data/Imdb_v1/test/neg/*.txt",
    })
    train_dataset = concatenate_datasets([add_label(dataset, "train_pos"),add_label(dataset, "train_neg")])
    test_dataset  = concatenate_datasets([add_label(dataset, "test_pos"),add_label(dataset, "test_neg")])

    small_train_dataset = train_dataset.train_test_split(test_size=0.9, shuffle=True, seed=42)['train']
    small_test_dataset = test_dataset.train_test_split(test_size=0.9, shuffle=True, seed=42)['test']


    return train_dataset, test_dataset
    #return small_train_dataset, small_test_dataset

# ---------------------------------------------------------------------

class DistillationTrainer(Trainer):
    def __init__(self, temperature=2.0,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        teacher_probs = inputs.pop("soft_label").to(get_device())

        outputs = model(**inputs)
        student_logits = outputs.logits

        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
        return (loss, outputs) if return_outputs else loss


def train_student_from_csv(soft_dataset, test_ds, model_name="distilbert-base-uncased", temperature=2.0):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True)
    soft_dataset = soft_dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3   # positive, neutral, negative
    )
    # Training arguments
    training_args = TrainingArguments(
        output_dir="distilbert-kd-softlabels",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        #weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        #logging_steps=50,
    )

    # Use custom distillation trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=temperature,
        args=training_args,
        train_dataset=soft_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    return student_model, tokenizer

def train_distilbert(train_data, test_data, model_name="distilbert-base-uncased"):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    small_tokenized_datasets = {
        "train": train_data.map(tokenize_function, batched=True),
        "test": test_data.map(tokenize_function, batched=True),
    }
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="distilbert-imdb-results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), # I have a gpu, might as well use it if possible
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_tokenized_datasets["train"],
        eval_dataset=small_tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    return model, tokenizer


def predict_sentiment(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
        
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=1).item()
    return "Positive" if pred == 1 else "Negative"

if __name__ == "__main__":
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    ds = load_soft_label_dataset("data/CSV FILE NAME HERE")

    model, tokenizer = train_student_from_csv(ds)

    start = time.time()

    print("Testing...")
    example = "This was terrible."
    print(predict_sentiment(model, tokenizer, example))

    elapsed = time.time() - start
    print(f"Time to process one sentiment: {elapsed}")
