
import pandas as pd
import numpy as np
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset


torch.manual_seed(42)

#Load and Rename
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")
test_df = pd.read_csv("test.csv")

# Rename columns to 'text' and 'label'
train_df.rename(columns={"comment": "text", "toxicity": "label"}, inplace=True)
valid_df.rename(columns={"comment": "text", "toxicity": "label"}, inplace=True)
test_df.rename(columns={"comment": "text"}, inplace=True)

# Preprocessing 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

train_df["clean_text"] = train_df["text"].apply(clean_text)
valid_df["clean_text"] = valid_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

#Naive Bayes with TF-IDF 
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = tfidf.fit_transform(train_df["clean_text"])
X_valid = tfidf.transform(valid_df["clean_text"])
X_test = tfidf.transform(test_df["clean_text"])

nb_model = MultinomialNB()
nb_model.fit(X_train, train_df["label"])
pred_valid_nb = nb_model.predict(X_valid)
pred_test_nb = nb_model.predict(X_test)

print("\nNaive Bayes Performance:")
print("Accuracy:", accuracy_score(valid_df["label"], pred_valid_nb))
print("F1-score:", f1_score(valid_df["label"], pred_valid_nb, average="weighted"))

# BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist())
valid_dataset = TextDataset(valid_df["text"].tolist(), valid_df["label"].tolist())
test_dataset = TextDataset(test_df["text"].tolist())  # test.csv has no labels

bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    seed=42,
    do_train=True,
    do_eval=True
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()

# Predict validation
pred_valid_bert = trainer.predict(valid_dataset).predictions
pred_valid_bert = np.argmax(pred_valid_bert, axis=1)

print("\nBERT Performance:")
print("Accuracy:", accuracy_score(valid_df["label"], pred_valid_bert))
print("F1-score:", f1_score(valid_df["label"], pred_valid_bert, average="weighted"))

# Predict test
pred_test_bert = trainer.predict(test_dataset).predictions
pred_test_bert = np.argmax(pred_test_bert, axis=1)

#  Save the file
test_output = test_df.copy()
test_output["out_label_model_Gen"] = pred_test_nb
test_output["out_label_model_Dis"] = pred_test_bert
test_output[["out_label_model_Gen", "out_label_model_Dis"]].to_csv("test.csv", index=False)

print("\nsampletest.csv generated .")
