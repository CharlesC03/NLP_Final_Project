# Trainer DistilBERT model
# Author: Daniel Alvarez

# Using the Stanford IMDB dataset.
# Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)
#
# Using the Equity Evaluation dataset
# Svetlana Kiritchenko, Saif Mohammad (2018)

import argparse
from datasets import load_dataset, concatenate_datasets,  DatasetDict
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


from Distilbert_Wrapper_Sub_Class import DistilbertWrapper

# DALVAREZ DEBUG FLAG:  I think these imports are Not used. SHOULD be safe to remove
from collections import Counter
import time
import sys
from datasets import Dataset

# helper functions
# -----------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


#get IMDB dataset, create train and test datasets.
def load_imdb_dataset(train_csv_path, test_csv_path):
    data = load_dataset(
        "csv",
        data_files={
            "train": train_csv_path,
            "test": test_csv_path
        }
    )
    return data["train"], data["test"]
# ====================================================

class DistillationTrainer(Trainer):
    def __init__(self, temperature=2.0,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.label_names = ["labels", "soft_label"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_probs = torch.Tensor(inputs.pop("soft_label")).to(model.device)

        outputs = model(**inputs)
        student_logits = outputs.logits

        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
        return (loss, outputs) if return_outputs else loss


# Trains a distilBERT model on teacher soft labels data.
# inputs:
#   train_csv_path(string): the path to a csv file containing the training dataset
#   test_csv_path(string): the path to a csv file containing the testing dataset
def train_student_from_csv(train_csv_path, test_csv_path=None, model_name="distilbert-base-uncased", temperature=2.0):
    # load data
    data = DatasetDict({
        "train": load_dataset("csv", data_files=train_csv_path)["train"]
    })
    if test_csv_path is not None:
        data["test"] = load_dataset("csv", data_files=test_csv_path)["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True)

    data = data.map(tokenize_fn, batched=True)
    # fix column names to match expected labels 
    def create_soft_label(example):
        example["soft_label"] = [example["label_0"], example["label_1"]]
        return example
    data = data.map(create_soft_label)
    data_collator = DataCollatorWithPadding(tokenizer)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels= 2   # negative, positive
    )
    # Training arguments
    training_args = TrainingArguments(
        output_dir="distilbert-kd-softlabels",
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        fp16=torch.cuda.is_available(), # if possible, use gpu to run faster
    )
    # Use custom distillation trainer
    trainer = DistillationTrainer(
        model=student_model,
        temperature=temperature,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    # save model
    student_model.save_pretrained("../models/distilbert_distilled_model")
    tokenizer.save_pretrained("../models/distilbert_distilled_model")
    
    return student_model, tokenizer

def train_distilbert_baseline(train_data, test_data, model_name="distilbert-base-uncased"):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(data):
        tokenized = tokenizer(data["text"], truncation=True, padding=False)
        tokenized["labels"] = data["label"]
        return tokenized
                                                                                               
    tokenized_datasets = {
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
        num_train_epochs=10,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), # if possible, use gpu to run faster
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    return model, tokenizer



# load the EEC dataset and clean string field values.
def load_eec_dataset():
    """
    Load the Equity Evaluation Corpus from HuggingFace.
    Returns HF Dataset dict: train / test (EEC has only one split, so we create one).
    """
    negative = ["anger", "fear","sadness"]
    positive = ["joy", "no emotion"]  
    ds = load_dataset("csv", data_files="../data/EEC/equity_evaluation_corpus.csv")
    ds = ds["train"]

    def normalize_fields_and_label(datum):
        datum["Gender"] = datum["Gender"].strip() if datum["Gender"] and datum["Gender"].strip() != "" else "no gender"
        datum["Race"] = datum["Race"].strip() if datum["Race"] and datum["Race"].strip() != "" else "no race"
        datum["Emotion"] = datum["Emotion"].strip() if datum["Emotion"] and datum["Emotion"].strip() != "" else "no emotion"
        if datum["Emotion"] in negative:
            datum["labels"] = 0
        elif datum["Emotion"] in positive:
            datum["labels"] = 1
        else:
            print(f"WARNING: Unexpected Emotion '{datum['Emotion']}' was found. Assigned 1 as default label.")
        return datum
    ds = ds.map(normalize_fields_and_label)
    split_ds = ds.train_test_split(test_size=0.2, seed=42)
    return split_ds["train"], split_ds["test"]

def generate_predictions(
        wrapper,
        dataset,
        text_column="sentence",
        output_path="default_output_path_predictions"
    ):
    
    results = []
    for example in tqdm(dataset, desc="Predicting sentiments"):
        text = example[text_column]
        probs = wrapper.predict(text)
        probs_np = probs.numpy()

        #testing reformatted csv output
        entry = {}
        for k, v in example.items():
            if k == "labels":
                entry["true_label"] = int(v)       # rename label → true_label
            elif k != "labels":
                entry[k] = v
                
        entry["label_0"]= float(probs_np[0])
        entry["label_1"] = float(probs_np[1])
        entry["pred_label"] = 0 if entry["label_0"] > entry["label_1"] else 1
        results.append(entry)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path + ".csv", index=False)
    df.to_parquet(output_path + ".parquet", index=False)
    print(f"Saved sentiment predictions → {output_path}.csv / .parquet")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with baseline or distilled IMDB models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "distilled"],
        required=True,
        help="Select which model pipeline to run: baseline or distilled."
    )
    parser.add_argument(
        "--train",
        type=str,
        default="True",
        help="Whether train a model. 'True' or 'False'. Default=True."
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="True",
        help="Whether to generate predictions. 'True' or 'False'. Default=True."
    )
    
    # check args
    args = parser.parse_args()
    run_train = args.train.lower() == "true"
    run_predict = args.predict.lower() == "true"

    global accuracy
    accuracy = evaluate.load("accuracy")

    # create wrapper used for predictions
    wrapper = DistilbertWrapper()

    # load datasets (IMDB and EEC) 
    train_imdb_ds, test_imdb_ds = load_imdb_dataset("../data/IMDB/imdb_train.csv", "../data/IMDB/imdb_test.csv")
    eec_train_ds, eec_test_ds = load_eec_dataset()

    # ==========================================================================
    #  Run on/ train baseline model
    # ==========================================================================
    if args.model == "baseline":
        print("\n=== Model selected: Baseline ===\n")
        if run_train:
            print("\n--- Training DistilBERT model on IMDB dataset ---")
            model, tokenizer = train_distilbert_baseline(train_imdb_ds, test_imdb_ds)
            model.save_pretrained("../models/distilbert_baseline_model")
            tokenizer.save_pretrained("../models/distilbert_baseline_model")
        if run_predict:
            print("\n--- Generating predictions with baseline DistilBERT model ---")
            wrapper.load_model("../models/distilbert_baseline_model")
            # generate predictions on IMDB dataset
            generate_predictions(
                wrapper,
                test_imdb_ds,
                text_column="text",
                output_path="../analysis/datasets/baseline_IMDB_predictions"
            )
            # generate predictions on EEC dataset
            generate_predictions(
                wrapper,
                concatenate_datasets([eec_train_ds, eec_test_ds]),
                text_column="Sentence",
                output_path="../analysis/datasets/baseline_EEC_predictions"
            )
            
    # ==========================================================================
    #  Run on/ train distilled model
    # ==========================================================================
    elif args.model == "distilled":
        print("\n=== Model selected: Baseline ===\n")
        # Train distilled model
        if run_train:
            print("\n--- Training distilled student model from teacher soft labels ---")
            model, tokenizer = train_student_from_csv(
                "../data/llama3.1/train.csv",
                "../data/llama3.1/test.csv"
            )
        if run_predict:
            print("\n--- Generating predictions with baseline DistilBERT model ---")
            wrapper.load_model("../models/distilbert_distilled_model")
            # generate predictions on IMDB dataset
            generate_predictions(
                wrapper,
                test_imdb_ds,
                text_column="text",
                output_path="../analysis/datasets/distilled_IMDB_predictions"
            )
            # generate predictions on EEC dataset
            generate_predictions(
                wrapper,
                concatenate_datasets([eec_train_ds, eec_test_ds]),
                text_column="Sentence",
                output_path="../analysis/datasets/distilled_EEC_predictions"
            )
