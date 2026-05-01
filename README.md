# Named Entity Recognition using BERT

## Project Overview

This project implements a **Named Entity Recognition (NER)** system using a **BERT-based transformer model** to identify and classify entities in text such as **Person, Organization, and Location**.

The model processes input text and outputs structured entity predictions with improved readability and confidence interpretation.

---

## Objective

* To build an NLP model capable of identifying entities in unstructured text
* To improve prediction reliability using **confidence calibration techniques**
* To enhance output interpretability through **entity grouping and labeling**

---

## Dataset Used

* **CoNLL-2003 Dataset**
* Contains annotated sentences with entity labels:

  * PER (Person)
  * ORG (Organization)
  * LOC (Location)
  * MISC (Miscellaneous)

---

## Technologies & Libraries

* Python
* Hugging Face Transformers
* PyTorch
* NumPy
* Streamlit (for deployment)

---

## Model Used

* **BERT (Bidirectional Encoder Representations from Transformers)**
* Fine-tuned for Named Entity Recognition task
* Token classification approach

---

## Key Features

### 1. Transformer-Based NER

* Uses pre-trained BERT model for high-quality contextual understanding

### 2. Confidence Calibration (New Feature)

* Implemented **temperature scaling** to adjust prediction confidence scores
* Helps reduce overconfident predictions and improves reliability

### 3. Entity Grouping

* Combines token-level outputs into meaningful full entities
* Example:
  `Barack + Obama → Barack Obama (Person)`

### 4. User-Friendly Output

* Converts raw model output (B-PER, I-PER) into readable format
* Improves interpretability for end users

### 5. Streamlit Web App

* Interactive interface to input text and visualize entity predictions

---

## Sample Output

Input:

```text
Barack Obama visited India.
```

Output:

```text
Barack Obama → Person  
India → Location
```


## ⭐ Acknowledgements

* Hugging Face Transformers
* CoNLL-2003 Dataset
* Open-source NLP community

---
