# LRCpredictor: Lactation Risk Classification Predictor

[![Web App](https://img.shields.io/badge/Web-Streamlit-red.svg)](https://lrcpredictor.streamlit.app/)

## Overview

**LRCpredictor** is a machine learning-based computational framework for predicting lactation risk categories of drugs based on molecular structure. Unlike existing models that predict indirect pharmacokinetic surrogates (e.g., milk-to-plasma ratios), LRCpredictor directly classifies drugs into clinically actionable risk categories following Dr. Thomas Hale's evidence-based Lactation Risk Categories (LRC) system.


---

## Key Features

- **Direct Clinical Prediction**: Classifies drugs into Low Risk (L1/L2) vs. High Risk (L4/L5) categories
- **Comprehensive Molecular Characterization**: 35 optimized descriptors from Mordred, RDKit, and MACCS
- **High Performance**: AUC = 0.8105, Accuracy = 74.68% on independent test set
- **Model Interpretability**: SHAP-based explanations identify molecular determinants of lactation risk
- **Structural Alerts**: 18 identified substructures associated with elevated risk
- **User-Friendly Web Platform**: https://lrcpredictor.streamlit.app/

---

## Repository Contents

```
LRCpredictor/LRCpredictor_Demo
├── README.md                              # This file
├── LRCpredictor_Demo.py                   # Demonstration script (Python)
├── gbdt_lactation_risk_pipeline.pkl       # Trained GBDT model
```

---
### Setup

#### Option 1: Using pip

```bash
# Install RDKit
conda install -c conda-forge rdkit

# Install other dependencies
pip install mordred scikit-learn shap pandas numpy matplotlib joblib
```

#### Option 2: Using environment.yml

```bash
conda env create -f environment.yml
conda activate lrcpredictor
```

---

## Quick Start

### 1. Load Required Files

Ensure the following files are in the same directory:
- `LRCpredictor_Demo.py` 
- `gbdt_lactation_risk_pipeline.pkl`

### 2. Run the Demo Script

#### Command Line

```
python LRCpredictor_Demo.py
```

### 3. Predict Your Own Drug

**Method 1: Modify the script**

Open `LRCpredictor_Demo.py` and locate the "CUSTOM PREDICTION" section (near the end):

```python
# MODIFY THESE TWO LINES:
custom_smiles = "YOUR_SMILES_HERE"
custom_drug_name = "YOUR_DRUG_NAME"
```

**Method 2: Use Python interactively**

```python
from LRCpredictor_Demo import *

# Load model
pipeline = load_trained_model('gbdt_lactation_risk_pipeline.pkl')
explainer = shap.TreeExplainer(pipeline['model'])

# Predict
predict_and_explain(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    drug_name="Aspirin",
    pipeline=pipeline,
    explainer=explainer
)
```

---

## Usage Examples

### Example 1: Amoxicillin (Low Risk)

```python
smiles = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
drug_name = "Amoxicillin"

predict_and_explain(smiles, drug_name, pipeline, explainer)
```

---

## Web Platform

Access the online prediction tool at: **https://lrcpredictor.streamlit.app/**

### Features:
- Single drug prediction from SMILES
- Batch prediction (CSV upload)
- Interactive SHAP visualizations
- Downloadable results
- No installation required

### Usage:
1. Visit the web platform
2. Enter drug name and SMILES (or upload CSV for batch prediction)
3. Click "Predict"
4. View results with SHAP interpretation
5. Download results if needed

---

**Last Updated**: December 2024  
**Version**: 1.0.0

For the latest updates, visit: https://github.com/Huangxiaojie2024/LRCpredictor
