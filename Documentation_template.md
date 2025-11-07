# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Epoch Engineers
**Team Members:** Shivam Sharma, Shail Kashyap, Anshuman Prakash, Mayank Raj 
**Submission Date:** 13 October 2025

---

## 1. Executive Summary
Our solution addresses the Smart Product Pricing Challenge by leveraging a multimodal learning approach that combines text and image embeddings to predict product prices. We used CLIP embeddings for both text and image features and applied a powerful ensemble of gradient boosting models (LightGBM + XGBoost + CatBoost) to improve accuracy and robustness.

## 2. Methodology Overview

### 2.1 Problem Analysis
We analyzed the relationship between product catalog content, product images, and their prices.

**Key Observations:**
1.Product descriptions (text) contained meaningful features such as brand, quantity, and attributes.
2.Image features captured additional signals like product type, packaging, and quality.
3.Price distribution was highly skewed, with outliers at the upper range.
4.A combination of textual and visual signals led to stronger predictive power than either alone.

### 2.2 Solution Strategy

**Approach Type:** Ensemble Model with Multimodal Inputs
**Core Innovation:** 
Using CLIP to generate 512-dimensional text embeddings and 512-dimensional image embeddings, then concatenating them with simple text statistics (text_len, num_words). Finally, we trained an ensemble of three gradient boosting models for better generalization and lower SMAPE.

## 3. Model Architecture

### 3.1 Architecture Overview
Product Catalog Content -----┐
                             │
                      CLIP Text Encoder ----> Text Embeddings
                             │
Product Images --------------┘
                             │
                      CLIP Image Encoder ----> Image Embeddings
                             │
                  Feature Concatenation (Text + Image + Extra Features)
                             │
                ┌────────────┼─────────────┐
                │            │             │
          n fold LightGBM n fold  XGBoost  n fold  CatBoost
                │            │             │
                └────── Weighted Ensemble ─┘
                             │
                      Final Price Prediction


### 3.2 Model Components

**Text Processing Pipeline:**
Preprocessing: Minimal cleaning (lowercasing, trimming)
Model: CLIP Text Encoder (openai/clip-vit-base-patch32)
Embedding Dim: 512

**Image Processing Pipeline:**
Preprocessing: Resize & normalization via CLIPProcessor
Model: CLIP Image Encoder (openai/clip-vit-base-patch32)
Embedding Dim: 512

**Feature Engineering:**
Text length and word count as extra numeric features
Outlier capping for extreme prices (top 1–2%)
---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** ~55.3 (on 75k subset), expected lower on full dataset after ensembling
- **Other Metrics:** RMSE tracked during LightGBM training (~0.77 on validation)


## 5. Conclusion
Our approach demonstrates how multimodal embeddings combined with gradient boosting ensemble models can effectively predict product prices at scale. CLIP encoders allowed us to capture rich semantic and visual cues, while LightGBM, XGBoost, and CatBoost provided strong tabular modeling capabilities. The method is scalable and can be further improved with outlier handling and hyperparameter tuning.

## Appendix

### A. Embeddings files
Drive Link: (https://drive.google.com/drive/folders/1qbW4O8mvxK-IvD_7OLrpKQXRJl24Rrl0?usp=sharing)


### B. Additional Results
Result on test data was SMAPE of 55.367.

---
