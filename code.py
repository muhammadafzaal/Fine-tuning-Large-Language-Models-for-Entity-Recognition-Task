from datasets import load_dataset
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import evaluate

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# List of entities
entity_categories = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL", "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD", "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH", "B-PLANT", "I-PLANT", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI"]

# True == Entity types not belonging to one of the following five will be set to zero:
#         - PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)
filtered_categories = True

# Since it is a sequnce eveluation task so we will use "seqeval" as metric
metric = evaluate.load("seqeval")

# PROBLEM: Tokens return by the tokenizer are longer than the list of categories/labels the dataset contains.
#          For example [CLS] and [SEP] tokens added by the tokenizer

# Below function solve this problem
def tokenize_alignment_categories(dataset, tokenizer, label_all_tokens=True):

  # Getting tokens from dataset
  input_tokens = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True)
  categories = []

  # Looping each category
  for i, category in enumerate(dataset["ner_tags"]):

      # Filltering some categories based on condition
      if(filtered_categories == True):
          for j, item in enumerate(category):
              if(item > 8) and not (13 <= item <= 14): category[j] = 0

      # Mapping to words
      word_ids = input_tokens.word_ids(batch_index=i)

      previous_word_idx = None
      category_ids = []

      # looping each mapping and assigning -100 if None
      for word_idx in word_ids:
          if word_idx is None:
              category_ids.append(-100)
          elif word_idx != previous_word_idx:
              category_ids.append(category[word_idx])
          else:
              category_ids.append(category[word_idx] if label_all_tokens else -100)
          previous_word_idx = word_idx

      # Adding to new categories list
      categories.append(category_ids)
  # Creating new key
  input_tokens["labels"] = categories
  return input_tokens

def compute_metrics(eval_preds):

  # eval_preds is a tuple that contains the predicted labels and the true labels
  logists, true_labels = eval_preds

  predicted_labels = np.argmax(logists, axis=2)

  true_categories = [
    [entity_categories[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predicted_labels, true_labels)
  ]

  prediction_categories = [
    [entity_categories[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predicted_labels, true_labels)
  ]
  # computing mteric
  results = metric.compute(predictions=prediction_categories, references=true_categories)
  return {
      "Accuracy": results["overall_accuracy"],
      "Precision": results["overall_precision"],
      "Recall": results["overall_recall"],
      "F1-Score": results["overall_f1"]
  }

# Filter out the non-English examples of the dataset
data_files = {"train": "train/train_en.jsonl", "test": "test/test_en.jsonl", "val": "val/val_en.jsonl"}
# Loading dataset
dataset = load_dataset("Babelscape/multinerd", data_files=data_files)

# Loading tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Transforming to avoid token and category/label alignment problem
transformed_dataset = dataset.map(tokenize_alignment_categories, batched=True, fn_kwargs={"tokenizer": tokenizer})


# Chosing BERT as LLM for this task
bert_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=31)
bert_model.to(device)

# Defining arguments for training
agruments = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01
)

# pads some model inputs into a batch to the lenght of the longest example and leads to faster tarining
data_collator = DataCollatorForTokenClassification(tokenizer)

# Defining Traininer
training = Trainer(
  bert_model,
  agruments,
  train_dataset=transformed_dataset["train"],
  eval_dataset=transformed_dataset["val"],
  data_collator=data_collator,
  tokenizer=tokenizer,
  compute_metrics=compute_metrics
)

# Start training
training.train()

# Saving model for future prediction
bert_model.save_pretrained("fine_tunned_bert_model")
tokenizer.save_pretrained("tokenizer_bert")



# Start testing on fune-tuned model
testing = training.predict(transformed_dataset["test"])
#results of testing
pd.DataFrame(testing[2].items(), columns=['Metric', 'Value'])
