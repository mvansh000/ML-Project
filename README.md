# Customer LTV Prediction

## Overview
This project predicts Customer Lifetime Value (LTV) using RFM analysis and XGBoost. It preprocesses retail data, clusters customers into Low, Mid, High LTV groups, and trains a model to predict these clusters.

## Structure
- `data/`: Datasets
- `notebooks/`: Jupyter notebooks
- `src/`: Python scripts
- `models/`: Saved models
- `docs/`: Documentation

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run `rfm_clustering.ipynb` to preprocess data.
3. Run `model_training.ipynb` to train and evaluate.

## Results
- Accuracy: 0.87
- F1 Scores: Low (0.90), Mid (0.61), High (0.50)
