# 🚀  AMAZON ML CHALLENGE PROJECT 


**GPU-Optimized Ensemble Training (n-Fold Each): LightGBM + XGBoost + CatBoost**

## 🏆 **Project Overview**

This solution was developed for the **Amazon ML Challenge 2025 – Product Pricing Task**.  
We implemented a **multi-modal ensemble pipeline** combining textual and visual data using **OpenAI’s CLIP model** with GPU-accelerated gradient boosting regressors (**LightGBM, XGBoost, CatBoost**).

Our approach integrates **image**, **text**, and **engineered features** to predict product prices with high robustness and accuracy.

---

## 🧩 **Dataset Preparation and Preprocessing**

### **Data Sources**

- **Amazon ML Challenge dataset:**
    - ~75,000 training products
    - ~75,000 test products
- **Each record includes:**
    - `catalog_content`: product title, description, pack quantity
    - `image_link`: product image URL

### **Feature Extraction**

- Used **CLIP (Vision Transformer)** for both images and text descriptions.
- Each modality produces a **512-dimensional embedding**, resulting in:<br>
  `512 (image)` + `512 (text)` = **1024 dimensions**
- **Concatenated** with engineered numeric features, such as:
    - **Description length**
    - **Word count**

### **Preprocessing Steps**

- **Cleaned text** (HTML removal, lowercasing, token counts).
- **Handled missing images** using zero vectors.
- **Normalized numeric features.**
- Used **CLIP’s tokenizer for text encoding** — no manual tokenization.
- Maintained the **official train/test split**.

---

## 🧠 **Model Architecture and Pipeline**

### **Multi-Modal Encoding**

- **Image Encoder:** CLIP ViT-B/32 → **512-D embedding**
- **Text Encoder:** CLIP Text Transformer → **512-D embedding**
- **Final Feature Vector:** 1024-D (image + text) + **numeric features**

### **Models**

- **LightGBM:** GPU-accelerated with `device=gpu`
- **XGBoost:** CUDA-based tree method (`tree_method=hist`, `device=cuda`)
- **CatBoost:** Trained using `task_type="GPU"`

### **Cross-Validation**

- Used **3-fold cross-validation** for each model to ensure robust and generalized predictions.

### **Ensembling**

- **Final prediction** = **Weighted Average** of LightGBM, XGBoost, and CatBoost outputs.
- **Equal or tuned weights** yielded the best SMAPE scores.

---

**Pipeline Summary:**

```
Image + Text → CLIP Encoders → 512-D each
+ Additional Features (text length, word count)
→ Concatenate to final feature vector (~1026+ dims)
→ LightGBM / XGBoost / CatBoost (GPU-enabled, 3-fold CV)
→ Weighted Ensemble → Final Price Prediction
```

---

## ⚙️ **Training Process and Validation**

### **Hardware & Acceleration**

- **Trained entirely on google collab T4 GPU.**
- GPU acceleration **drastically reduced training time** for the 75K-sample dataset.

### **Hyperparameter Tuning**

- Parameters such as **learning rate, tree depth, and number of leaves** were tuned via grid search.
- **Early stopping** based on validation SMAPE to prevent overfitting.

### **Validation Metric: SMAPE**

The **Symmetric Mean Absolute Percentage Error (SMAPE)** was used:

**Symmetric Mean Absolute Percentage Error (SMAPE):**

```
           |P_pred - P_actual|
SMAPE = ---------------------------- × 100%
        (|P_pred| + |P_actual|) / 2
```

Where:
- **P_pred** = Predicted price
- **P_actual** = Actual price

- Ranges from **0% (perfect)** to **200% (worst)**
- Model weights were optimized to **minimize SMAPE on validation folds**

### **Final Model**

- After cross-validation, **each model was retrained on the entire training set** (or aggregated folds).
- The **ensemble was applied on the withheld test set** to generate final predictions.

---

## 📊 **Final Results**

| Metric                | Value          | Description                                   |
|-----------------------|---------------|-----------------------------------------------|
| **SMAPE (Private Test)**     | ≈ **55.3 as score on test data**   | Lower is better; demonstrates strong generalization |
| **Training Time**           | Reduced by ~**60%** | Due to full GPU acceleration                  |
| **Best Performing Ensemble**| LightGBM + XGBoost + CatBoost | Weighted average combination         |

> The **GPU-optimized multi-modal ensemble** significantly outperformed simple baselines in both accuracy and robustness.

---

## 🧰 **Setup Instructions**

### **Environment**

- **Python 3.8+**
- **CUDA-enabled PyTorch installation**

### **Dependencies**

Install via:

```sh
pip install -r requirements.txt
```

**Key Packages:**

- `torch`, `transformers` – CLIP model
- `pandas`, `numpy`, `scikit-learn` – data handling & CV
- `lightgbm[gpu]`, `xgboost`, `catboost` – GPU regressors
- `tqdm`, `matplotlib`, `seaborn` – visualization (optional)

### **Data Setup**

- Place datasets in `data/` directory (`train.csv`, `test.csv`)
- Place downloaded product images in `images/` directory (keyed by sample ID)

---

### **Feature Extraction**

```sh
Embeddings available to download on : (Drive Link) : https://drive.google.com/drive/folders/1qbW4O8mvxK-IvD_7OLrpKQXRJl24Rrl0?usp=sharing
```

- Use the above or make your own embeddings from text and image **512-D embeddings** for all text and image samples.

---

### **Training**

```sh
python code.py
```

- Performs:
    - **3-fold CV**
    - **Training of all models**
    - **Ensemble prediction generation**

---

- Computes **SMAPE** between predicted and actual prices.

---

## 👨‍💻 **Team Members**

| Name         |
|--------------|
| **Shivam Sharma**   | 
| **Mayank Raj**   | 
| **Shail Kashyap**     |
| **Anshuman Prakash**   |

**Acknowledgments:**

- **Hugging Face team** for CLIP model
- **Amazon ML Challenge organizers** for dataset and framework
- **Open-source contributors** to PyTorch, LightGBM, XGBoost, and CatBoost

---
