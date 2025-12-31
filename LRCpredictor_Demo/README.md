# LRCpredictor: Lactation Risk Classification Predictor

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)
[![Web App](https://img.shields.io/badge/Web-Streamlit-red.svg)](https://lrcpredictor.streamlit.app/)

## Overview

**LRCpredictor** is a machine learning-based computational framework for predicting lactation risk categories of drugs based on molecular structure. Unlike existing models that predict indirect pharmacokinetic surrogates (e.g., milk-to-plasma ratios), LRCpredictor directly classifies drugs into clinically actionable risk categories following Dr. Thomas Hale's evidence-based Lactation Risk Categories (LRC) system.

This repository contains the supplementary materials for the manuscript:

> **Machine Learning-Based Prediction of Drug Lactation Risk: Bridging Molecular Features and Breastfeeding Safety**  
> Peineng Liu, Shaokai Huang, Xiaochun Xie, Jiajia Chen, Shanshan Wu, Lina Huang, Xiaojie Huang*  
> *Department of Pharmacy, Jieyang People's Hospital, Jieyang 522000, China*  
> Correspondence: huangxj46@alumni.sysu.edu.cn

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
LRCpredictor/
├── README.md                              # This file
├── LRCpredictor_Demo.py                   # Demonstration script (Python)
├── LRCpredictor_Demo.ipynb                # Demonstration notebook (Jupyter)
├── gbdt_lactation_risk_pipeline.pkl       # Trained GBDT model
├── dataset/
│   ├── LRC_dataset.sdf                    # Complete dataset (SDF format)
│   ├── training_set.csv                   # Training set (314 drugs)
│   └── test_set.csv                       # Test set (78 drugs)
├── descriptors/
│   ├── Mordred.csv                        # Mordred descriptors
│   ├── RDKit.csv                          # RDKit descriptors
│   ├── MACCS.csv                          # MACCS fingerprints
│   └── Mordred+RDKit+MACCS.csv           # Combined features
├── models/
│   ├── RF_model.pkl                       # Random Forest model
│   ├── GBDT_model.pkl                     # Gradient Boosting model (optimal)
│   ├── AdaBoost_model.pkl                 # AdaBoost model
│   ├── LightGBM_model.pkl                 # LightGBM model
│   └── XGBoost_model.pkl                  # XGBoost model
├── results/
│   ├── feature_selection_results.csv      # Feature selection performance
│   ├── model_comparison.csv               # Algorithm comparison
│   ├── shap_summary.png                   # SHAP summary plot
│   └── structural_alerts.csv              # Identified structural alerts
└── LICENSE                                 # License information
```

---

## Model Information

### Algorithm
- **Model Type**: Gradient Boosting Decision Tree (GBDT)
- **Feature Selection**: Embedded Tree-based (ETB) method
- **Optimization**: Bayesian optimization with Tree-structured Parzen Estimator

### Dataset
- **Total Drugs**: 392 compounds
  - Low Risk (L1/L2): 211 drugs
  - High Risk (L4/L5): 181 drugs
- **Training Set**: 314 drugs (80%)
- **Test Set**: 78 drugs (20%)
- **Data Source**: Hale's Medications & Mothers' Milk 2025-2026 Edition

### Features
- **Total Descriptors**: 35 molecular features
  - **Mordred 2D**: Topological indices, autocorrelation descriptors, BCUT indices
  - **RDKit**: Physicochemical properties, connectivity indices, drug-likeness
  - **MACCS**: Structural fingerprint patterns
- **Feature Categories**: Electronic properties, topological indices, polarizability, drug-likeness

### Performance Metrics (Test Set)

| Metric | Value |
|--------|-------|
| AUC | 0.8105 |
| Accuracy | 0.7468 (74.68%) |
| Sensitivity | 0.7222 (72.22%) |
| Specificity | 0.7619 (76.19%) |
| MCC | 0.4516 |

**Extreme Category Performance (L1 vs. L5)**:
- AUC: 0.9000
- Accuracy: 0.8519 (85.19%)

---

## Installation

### Requirements

```bash
Python >= 3.7
RDKit >= 2022.03.1
Mordred >= 1.2.0
scikit-learn >= 1.0.0
SHAP >= 0.41.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
joblib >= 1.0.0
```

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
- `LRCpredictor_Demo.py` (or `LRCpredictor_Demo.ipynb`)
- `gbdt_lactation_risk_pipeline.pkl`

### 2. Run the Demo Script

#### Command Line

```bash
python LRCpredictor_Demo.py
```

#### Jupyter Notebook

```bash
jupyter notebook LRCpredictor_Demo.ipynb
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

**Output:**
```
PREDICTION RESULTS
==================
Drug Name:              Amoxicillin
Predicted Risk Level:   Low Risk (L1/L2)
Low Risk Probability:   0.9245 (92.45%)
High Risk Probability:  0.0755 (7.55%)
```

### Example 2: Amiodarone (High Risk)

```python
smiles = "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1"
drug_name = "Amiodarone"

predict_and_explain(smiles, drug_name, pipeline, explainer)
```

**Output:**
```
PREDICTION RESULTS
==================
Drug Name:              Amiodarone
Predicted Risk Level:   High Risk (L4/L5)
Low Risk Probability:   0.2156 (21.56%)
High Risk Probability:  0.7844 (78.44%)
```

### Example SMILES Codes

| Drug | Category | SMILES |
|------|----------|--------|
| Aspirin | Low Risk | `CC(=O)Oc1ccccc1C(=O)O` |
| Ibuprofen | Low Risk | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |
| Caffeine | Low Risk | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` |
| Warfarin | High Risk | `CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O` |
| Methotrexate | High Risk | `CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O` |

---

## Output Interpretation

### 1. Risk Classification

- **Low Risk (L1/L2)**: Generally compatible with breastfeeding
  - L1: Safest - Extensive data showing no adverse effects
  - L2: Safer - Limited data, but no documented adverse effects

- **High Risk (L4/L5)**: Use with caution or contraindicated
  - L4: Possibly hazardous - Evidence of risk or theoretical concern
  - L5: Contraindicated - Significant documented risk or cytotoxic agents

### 2. Probability Scores

- Values range from 0.0 to 1.0 (0% to 100%)
- Higher confidence indicated by probabilities closer to extremes
- Probabilities near 0.5 suggest borderline cases requiring clinical judgment

### 3. SHAP Force Plot

![SHAP Interpretation](docs/shap_explanation.png)

**How to read the force plot:**
- **Red arrows (→)**: Features pushing toward HIGH RISK
- **Blue arrows (←)**: Features pushing toward LOW RISK
- **Arrow length**: Magnitude of contribution
- **Base value**: Average prediction across training data
- **Output value**: Final predicted probability

**Key molecular determinants identified:**
1. Electronic properties (VSA_EState1, VSA_EState2)
2. Topological indices (IC1, MIC1, ABC)
3. Autocorrelation descriptors (AATS, ATSC, GATS)
4. Polarizability indices (BCUT descriptors)
5. Drug-likeness (qed, HallKierAlpha)

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

## Dataset Description

### Lactation Risk Categories (LRC)

The LRC system classifies medications into five categories:

| Category | Risk Level | Description |
|----------|------------|-------------|
| **L1** | Safest | Extensive data in nursing mothers with no adverse effects |
| **L2** | Safer | Limited studies showing no adverse effects |
| **L3** | Moderately Safe | No controlled studies; theoretical concerns |
| **L4** | Possibly Hazardous | Positive evidence of risk |
| **L5** | Contraindicated | Significant risk documented |

**Note**: L3 drugs were excluded from modeling as they represent an intermediate category requiring individual clinical assessment.

### Data Source

Hale TW. *Hale's Medications & Mothers' Milk 2025-2026: A Manual of Lactational Pharmacology*. 21st Edition. New York: Springer Publishing Company; 2024.

### Dataset Statistics

| Property | Training Set | Test Set | Combined |
|----------|--------------|----------|----------|
| Total Drugs | 314 | 78 | 392 |
| Low Risk (L1/L2) | 169 | 42 | 211 |
| High Risk (L4/L5) | 145 | 36 | 181 |
| Molecular Weight | 100-900 Da | 100-850 Da | 100-900 Da |
| LogP Range | -4 to +10 | -3 to +9 | -4 to +10 |

---

## Model Development

### Feature Selection

Four complementary strategies were evaluated:
1. **ETB (Embedded Tree-based)** - Selected as optimal
2. **RFECV (Recursive Feature Elimination)**
3. **mRMR (minimum Redundancy Maximum Relevance)**
4. **SFM (SelectFromModel)**

**Optimal Result**: ETB selected 35 features achieving best performance

### Algorithm Comparison

Five ensemble algorithms were systematically evaluated:

| Algorithm | AUC | Accuracy | MCC |
|-----------|-----|----------|-----|
| **GBDT** | **0.8105** | **0.7468** | **0.4516** |
| LightGBM | 0.8021 | 0.7468 | 0.4475 |
| XGBoost | 0.7939 | 0.7468 | 0.4478 |
| AdaBoost | 0.7911 | 0.7597 | 0.4774 |
| Random Forest | 0.7823 | 0.7468 | 0.4495 |

**GBDT was selected as the final model** based on highest AUC and balanced performance.

### Hyperparameter Optimization

- **Method**: Bayesian optimization with TPE algorithm
- **Framework**: Hyperopt library
- **Validation**: 10-fold cross-validation
- **Metric**: Matthews Correlation Coefficient (MCC)

---

## Structural Alerts

The study identified **18 structural alerts** statistically associated with lactation risk:

### High-Risk Exclusive Alerts (5)

1. Aromatic nitrogen heterocycles with electron-withdrawing groups
2. Halogenated aromatic systems (especially iodine-containing)
3. Sulfonamide functional groups
4. Nitro-aromatic compounds
5. Complex polycyclic systems with quaternary carbons

### Common Alerts (13)

Including aromatic rings, aliphatic chains, carbonyl groups, etc.

**Statistical Metrics**:
- Likelihood Ratio > 1.5
- Prevalence in high-risk drugs > 30%
- Minimum occurrence ≥ 5 compounds

---

## Applicability Domain

The model's applicability domain was assessed using Euclidean distance analysis:

- **Training Set Coverage**: 100% (by definition)
- **Test Set Coverage**: 97.4% (76/78 drugs)
- **Threshold**: Normalized distance ≤ 1.0

**Recommendation**: Predictions for compounds outside the applicability domain should be interpreted with caution.

---

## Citation

If you use LRCpredictor in your research, please cite:

```bibtex
@article{liu2024lrcpredictor,
  title={Machine Learning-Based Prediction of Drug Lactation Risk: Bridging Molecular Features and Breastfeeding Safety},
  author={Liu, Peineng and Huang, Shaokai and Xie, Xiaochun and Chen, Jiajia and Wu, Shanshan and Huang, Lina and Huang, Xiaojie},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

**Important**: LRCpredictor is intended for **research purposes only** and should not replace clinical judgment. Lactation risk assessments should be made by qualified healthcare professionals considering:

- Individual patient factors
- Drug dosage and duration
- Infant age and health status
- Alternative medication options
- Benefit-risk assessment

**For clinical decisions, always consult:**
- Dr. Thomas Hale's *Medications & Mothers' Milk*
- LactMed database (NIH/NLM)
- Clinical pharmacist or lactation consultant

---

## Contact

**Corresponding Author**: Xiaojie Huang  
**Email**: huangxj46@alumni.sysu.edu.cn  
**Institution**: Department of Pharmacy, Jieyang People's Hospital  
**Address**: Jieyang 522000, Guangdong Province, China

**GitHub Issues**: For technical questions or bug reports, please open an issue on our [GitHub repository](https://github.com/Huangxiaojie2024/LRCpredictor/issues).

---

## Acknowledgments

- Dr. Thomas W. Hale for the LRC system
- RDKit and Mordred developers for molecular descriptor tools
- The SHAP library for model interpretability
- Streamlit for web platform development
- All contributors and users of LRCpredictor

---

## References

1. Hale TW. *Hale's Medications & Mothers' Milk 2025-2026*. 21st ed. Springer Publishing Company; 2024.
2. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst*. 2017;30:4765-4774.
3. Moriwaki H, Tian YS, Kawashita N, Takagi T. Mordred: a molecular descriptor calculator. *J Cheminform*. 2018;10:4.
4. Landrum G. RDKit: Open-source cheminformatics. http://www.rdkit.org

---

**Last Updated**: December 2024  
**Version**: 1.0.0

For the latest updates, visit: https://github.com/Huangxiaojie2024/LRCpredictor
