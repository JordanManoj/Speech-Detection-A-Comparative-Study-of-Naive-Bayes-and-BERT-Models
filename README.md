# Speech-Detection-A-Comparative-Study-of-Naive-Bayes-and-BERT-Models

##  Overview
This project detects **toxic language** in online comments using two different approaches:
1. **Naive Bayes** – a traditional machine learning model that works well with TF-IDF(Term Frequency-Inverse Document Frequency) text features.
2. **BERT** – a deep learning transformer model that understands the context of words.

The aim is to compare the speed, accuracy, and usefulness of both models for content moderation tasks.  
Toxic speech here refers to comments that are **rude, abusive, hateful, or offensive**.  

This kind of system can be used in:
- Social media moderation
- Online gaming chat monitoring
- Community discussion forums

---

## Dataset
We have three CSV files:
- **train.csv** – used to train the models (has `comment` and `toxicity` labels).
- **valid.csv** – used to check model performance (has labels).
- **test.csv** – used to make final predictions (no labels).

**Columns:**
- `comment` → The actual text.
- `toxicity` → Target label:
  - `0` = Not toxic
  - `1` = Toxic
- `comment_id`, `split` → Metadata.

---

##  Project Steps
1. **Data Cleaning**
   - Convert text to lowercase.
   - Remove punctuation and special characters.
   - Remove stopwords (for Naive Bayes).
   - Tokenize words.
   - Use:
     - **TF-IDF** for Naive Bayes.
     - **BERT tokenizer** for BERT.

2. **Model Training**
   - **Naive Bayes**: Simple, fast, works well for basic text classification.
   - **BERT**: More powerful, understands context, but slower.

3. **Evaluation**
   - Metrics:
     - Accuracy
     - Weighted F1-score
   - Compare both models on the validation set.

4. **Prediction**
   - Use both models to predict labels for `test.csv`.
   - Save predictions in two columns:
     - `out_label_model_Gen` (Naive Bayes)
     - `out_label_model_Dis` (BERT)

---

## Results
| Model        | Accuracy | F1-score   |
|--------------|----------|------------|
| Naive Bayes  | 0.8667   | 0.8086     |
| BERT         | 0.8688   | 0.8078     |

**Key Points:**
- Naive Bayes is much faster but less accurate.
- BERT captures subtle meaning and context, giving higher accuracy.

* Here the `test.csv` file will be replaced with `sampletest.csv`. 
