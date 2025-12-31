#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
LRCpredictor: Lactation Risk Classification Predictor
==============================================================================

Supplementary Code for Manuscript:
"Machine Learning-Based Prediction of Drug Lactation Risk: Bridging Molecular 
Features and Breastfeeding Safety"

Authors: Peineng Liu, Shaokai Huang, Xiaochun Xie, Jiajia Chen, Shanshan Wu, 
         Lina Huang, Xiaojie Huang*
Affiliation: Department of Pharmacy, Jieyang People's Hospital, Jieyang, China
Correspondence: huangxj46@alumni.sysu.edu.cn

This demonstration script illustrates the usage of the trained LRCpredictor 
model for predicting lactation risk categories of drugs from their molecular 
SMILES structures.

Model Information:
------------------
- Algorithm: Gradient Boosting Decision Tree (GBDT)
- Features: 35 molecular descriptors (Mordred + RDKit + MACCS)
- Feature Selection: Embedded Tree-based (ETB) method
- Training Set: 314 drugs (L1/L2: 169, L4/L5: 145)
- Test Set: 78 drugs (L1/L2: 42, L4/L5: 36)
- Performance: AUC = 0.8105, ACC = 0.7468, SEN = 0.7222, SPE = 0.7619

Input:
------
SMILES (Simplified Molecular Input Line Entry System) notation

Output:
-------
- Binary classification: Low Risk (L1/L2) or High Risk (L4/L5)
- Probability scores for both risk categories
- SHAP force plot for model interpretation

Web Platform:
-------------
https://lrcpredictor.streamlit.app/

GitHub Repository:
------------------
https://github.com/Huangxiaojie2024/LRCpredictor

==============================================================================
"""

# Import required libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LRCpredictor")
print("=" * 80)
print("✓ Libraries imported successfully\n")


# ==============================================================================
# SECTION 1: Molecular Descriptor Calculation
# ==============================================================================

def calculate_molecular_descriptors(smiles, drug_name="Unknown"):
    """
    Calculate 35 molecular descriptors from SMILES structure.
    
    This function computes a comprehensive set of molecular descriptors that 
    encode physicochemical properties, electronic characteristics, topological 
    indices, and structural patterns. The descriptors were selected through 
    systematic feature selection (Embedded Tree-based method) from over 2000 
    initial features.
    
    Feature Categories:
    -------------------
    1. Mordred 2D descriptors: Topological indices (ABC, SpMax_A, etc.), 
       autocorrelation descriptors (AATS, ATSC, GATS), BCUT descriptors
    2. RDKit descriptors: Physicochemical properties (VSA_EState, qed), 
       topological descriptors (IC1, MIC1, JGI3), connectivity indices
    
    The 35 selected features represent the optimal balance between model 
    performance and interpretability as identified in our study.
    
    Parameters
    ----------
    smiles : str
        SMILES notation of the drug molecule
    drug_name : str, optional
        Name of the drug for labeling (default: "Unknown")
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing 35 molecular descriptors, or None if calculation fails
    
    Notes
    -----
    - Invalid SMILES will return None
    - Missing or invalid descriptor values are set to 0.0
    - Descriptors are calculated using RDKit and Mordred libraries
    """
    try:
        # Parse SMILES to molecular object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"✗ Error: Invalid SMILES for {drug_name}")
            return None
        
        # Calculate Mordred 2D descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        desc_values = calc(mol)
        
        # Store Mordred descriptors
        mordred_dict = {}
        for desc_name, value in zip(calc.descriptors, desc_values):
            try:
                if isinstance(value, (int, float, np.number)):
                    if not (np.isnan(value) or np.isinf(value)):
                        mordred_dict[str(desc_name)] = float(value)
                    else:
                        mordred_dict[str(desc_name)] = 0.0
                else:
                    mordred_dict[str(desc_name)] = 0.0
            except:
                mordred_dict[str(desc_name)] = 0.0
        
        # Calculate RDKit descriptors
        rdkit_dict = {}
        rdkit_desc_list = Descriptors._descList
        for desc_name, desc_func in rdkit_desc_list:
            try:
                value = desc_func(mol)
                rdkit_dict[desc_name] = 0.0 if pd.isna(value) else float(value)
            except:
                rdkit_dict[desc_name] = 0.0
        
        # Merge descriptor dictionaries
        all_descriptors = {**mordred_dict, **rdkit_dict}
        
        # Extract the 35 features selected by ETB method
        required_features = [
            'ABC', 'SpMax_A', 'VE1_A', 'VR1_A', 'AATS1dv', 'AATS3v', 'AATS3i',
            'ATSC2dv', 'ATSC5Z', 'ATSC4se', 'ATSC5p', 'ATSC6i', 'ATSC7i',
            'AATSC2d', 'AATSC1i', 'GATS1dv', 'GATS3dv', 'GATS3s', 'GATS2i',
            'BCUTc-1h', 'BCUTse-1l', 'BCUTi-1h', 'BCUTi-1l', 'C3SP2',
            'AXp-1d', 'AETA_eta_F', 'IC1', 'MIC1', 'VSA_EState1', 'VSA_EState2',
            'JGI3', 'MaxEStateIndex', 'qed', 'BCUT2D_MRLOW', 'HallKierAlpha'
        ]
        
        # Create descriptor DataFrame
        selected_descriptors = {
            feat: all_descriptors.get(feat, 0.0) for feat in required_features
        }
        
        descriptor_df = pd.DataFrame(
            [selected_descriptors], 
            columns=required_features, 
            index=[drug_name]
        )
        
        return descriptor_df
        
    except Exception as e:
        print(f"✗ Error calculating descriptors: {str(e)}")
        return None


# ==============================================================================
# SECTION 2: Model Loading
# ==============================================================================

def load_trained_model(model_path='gbdt_lactation_risk_pipeline.pkl'):
    """
    Load the pre-trained GBDT model pipeline.
    
    The pipeline includes:
    - Trained GBDT classifier (optimized via Bayesian optimization)
    - StandardScaler for feature normalization
    - Feature names (35 descriptors)
    
    Model Performance Metrics (Test Set):
    --------------------------------------
    - AUC: 0.8105
    - Accuracy: 0.7468
    - Sensitivity: 0.7222
    - Specificity: 0.7619
    - MCC: 0.4516
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file
    
    Returns
    -------
    dict or None
        Dictionary containing 'model', 'scaler', and 'feature_names', 
        or None if loading fails
    """
    try:
        pipeline = joblib.load(model_path)
        print(f"✓ Model loaded successfully: {model_path}")
        print(f"  Algorithm: {type(pipeline['model']).__name__}")
        print(f"  Features: {len(pipeline['feature_names'])} descriptors\n")
        return pipeline
    except FileNotFoundError:
        print(f"✗ Error: Model file '{model_path}' not found!")
        print("  Please ensure the .pkl file is in the same directory.\n")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}\n")
        return None


# ==============================================================================
# SECTION 3: Risk Prediction
# ==============================================================================

def predict_lactation_risk(descriptor_df, pipeline):
    """
    Predict lactation risk category using the trained GBDT model.
    
    The model classifies drugs into two categories based on Hale's Lactation 
    Risk Categories (LRC) system:
    - Low Risk: L1 (safest) and L2 (safer)
    - High Risk: L4 (possibly hazardous) and L5 (contraindicated)
    
    Note: L3 (moderately safe) drugs were excluded from training as they 
    represent an intermediate category requiring individual assessment.
    
    Parameters
    ----------
    descriptor_df : pd.DataFrame
        Calculated molecular descriptors
    pipeline : dict
        Loaded model pipeline
    
    Returns
    -------
    tuple
        (results DataFrame, scaled descriptors DataFrame)
        - results: Contains drug name, prediction, risk level, and probabilities
        - scaled_descriptors: Standardized descriptor values used for prediction
        Returns (None, None) if prediction fails
    """
    try:
        model = pipeline['model']
        scaler = pipeline['scaler']
        feature_names = pipeline['feature_names']
        
        # Ensure correct feature order
        descriptor_df = descriptor_df[feature_names]
        
        # Standardize features using training set parameters
        descriptor_scaled = pd.DataFrame(
            scaler.transform(descriptor_df),
            columns=feature_names,
            index=descriptor_df.index
        )
        
        # Generate predictions
        predictions = model.predict(descriptor_scaled)
        probabilities = model.predict_proba(descriptor_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Drug Name': descriptor_df.index,
            'Prediction': predictions,
            'Risk Level': ['High Risk (L4/L5)' if p == 1 else 'Low Risk (L1/L2)' 
                          for p in predictions],
            'Low Risk Probability': probabilities[:, 0],
            'High Risk Probability': probabilities[:, 1]
        })
        
        return results, descriptor_scaled
    
    except Exception as e:
        print(f"✗ Prediction error: {str(e)}")
        return None, None


# ==============================================================================
# SECTION 4: SHAP Interpretation
# ==============================================================================

def generate_shap_explanation(explainer, descriptor_original, descriptor_scaled, 
                              pipeline, prediction_prob):
    """
    Generate SHAP force plot for model interpretation.
    
    SHAP (SHapley Additive exPlanations) values provide feature-level 
    explanations by quantifying each descriptor's contribution to the 
    prediction. The force plot visualizes how features push the prediction 
    from the base value (population average) toward the final output.
    
    Interpretation Guide:
    ---------------------
    - Red features: Increase high risk probability
    - Blue features: Decrease high risk probability (increase low risk)
    - Feature magnitude: Strength of contribution
    - Base value: Average model prediction across training data
    
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object
    descriptor_original : pd.DataFrame
        Original (unscaled) descriptor values for display
    descriptor_scaled : pd.DataFrame
        Scaled descriptor values for SHAP calculation
    pipeline : dict
        Model pipeline
    prediction_prob : float
        Predicted high risk probability
    
    Returns
    -------
    matplotlib.figure.Figure or None
        SHAP force plot figure, or None if generation fails
    """
    try:
        # Calculate SHAP values
        shap_values = explainer.shap_values(descriptor_scaled.values)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values_high_risk = shap_values[1][0]  # Class 1 (High Risk)
        else:
            shap_values_high_risk = shap_values[0]
        
        # Get base value (expected value)
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Convert from logit space to probability space
        base_value_prob = 1 / (1 + np.exp(-expected_value))
        shap_sum = np.sum(shap_values_high_risk)
        
        # Scale SHAP values to probability space
        if abs(shap_sum) > 1e-10:
            scaling_factor = (prediction_prob - base_value_prob) / shap_sum
            shap_values_prob = shap_values_high_risk * scaling_factor
        else:
            shap_values_prob = shap_values_high_risk
        
        # Extract feature information
        feature_names = pipeline['feature_names']
        feature_values = descriptor_original.values[0]
        
        # Generate SHAP force plot
        shap.force_plot(
            base_value_prob,
            shap_values_prob,
            features=feature_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            link='identity'
        )
        
        plt.title(f"SHAP Force Plot - {descriptor_original.index[0]}", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("High Risk Probability", fontsize=12)
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"✗ Error generating SHAP plot: {str(e)}")
        return None


# ==============================================================================
# SECTION 5: Complete Prediction Workflow
# ==============================================================================

def predict_and_explain(smiles, drug_name, pipeline, explainer, 
                        show_descriptors=False, save_figures=False):
    """
    Execute complete prediction workflow with visualization.
    
    This function orchestrates the entire prediction process:
    1. Calculate molecular descriptors from SMILES
    2. Predict lactation risk category
    3. Generate probability visualization
    4. Create SHAP explanation plot
    5. Optionally display descriptor values
    
    Parameters
    ----------
    smiles : str
        SMILES notation of the drug
    drug_name : str
        Name of the drug
    pipeline : dict
        Loaded model pipeline
    explainer : shap.TreeExplainer
        SHAP explainer object
    show_descriptors : bool, optional
        Display molecular descriptor values (default: False)
    save_figures : bool, optional
        Save plots as PNG files (default: False)
    
    Returns
    -------
    None
        Results are printed and visualized
    """
    print("=" * 80)
    print(f"Lactation Risk Prediction: {drug_name}")
    print("=" * 80)
    print(f"SMILES: {smiles}\n")
    
    # Step 1: Calculate descriptors
    print("[1/3] Calculating molecular descriptors...")
    descriptors = calculate_molecular_descriptors(smiles, drug_name)
    
    if descriptors is None:
        print("✗ Prediction failed: Invalid SMILES structure\n")
        return
    
    print(f"✓ Successfully calculated {len(descriptors.columns)} descriptors\n")
    
    # Step 2: Predict risk
    print("[2/3] Predicting lactation risk category...")
    results, descriptors_scaled = predict_lactation_risk(descriptors, pipeline)
    
    if results is None:
        print("✗ Prediction failed\n")
        return
    
    # Extract results
    risk_level = results.iloc[0]['Risk Level']
    low_prob = results.iloc[0]['Low Risk Probability']
    high_prob = results.iloc[0]['High Risk Probability']
    
    # Display prediction results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"Drug Name:              {drug_name}")
    print(f"Predicted Risk Level:   {risk_level}")
    print(f"Low Risk Probability:   {low_prob:.4f} ({low_prob*100:.2f}%)")
    print(f"High Risk Probability:  {high_prob:.4f} ({high_prob*100:.2f}%)")
    print("=" * 80 + "\n")
    
    # Visualize probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Low Risk\n(L1/L2)', 'High Risk\n(L4/L5)']
    probabilities = [low_prob, high_prob]
    colors = ['#4CAF50', '#F44336']
    
    bars = ax.bar(categories, probabilities, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax.set_title(f'Lactation Risk Prediction: {drug_name}', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    # Add probability labels
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.1%}', ha='center', va='bottom', 
               fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    if save_figures:
        plt.savefig(f'{drug_name}_probability.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {drug_name}_probability.png")
    plt.show()
    
    # Step 3: Generate SHAP explanation
    print("[3/3] Generating SHAP interpretation...\n")
    shap_fig = generate_shap_explanation(
        explainer, descriptors, descriptors_scaled, pipeline, high_prob
    )
    
    if shap_fig is not None:
        if save_figures:
            plt.savefig(f'{drug_name}_SHAP.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {drug_name}_SHAP.png")
        plt.show()
        
        print("\n" + "=" * 80)
        print("SHAP INTERPRETATION GUIDE")
        print("=" * 80)
        print("• Red arrows (→):  Features increasing HIGH RISK probability")
        print("• Blue arrows (←): Features decreasing HIGH RISK probability")
        print("• Arrow length:    Magnitude of feature contribution")
        print("• Base value:      Average prediction across training data")
        print("• Output value:    Final predicted probability for this drug")
        print("=" * 80 + "\n")
    
    # Optional: Display descriptor values
    if show_descriptors:
        print("=" * 80)
        print("MOLECULAR DESCRIPTORS")
        print("=" * 80)
        descriptor_table = pd.DataFrame({
            'Descriptor': descriptors.columns,
            'Value': descriptors.values[0],
            'Scaled Value': descriptors_scaled.values[0]
        })
        print(descriptor_table.to_string(index=False))
        print("=" * 80 + "\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "█" * 80)
    print("INITIALIZATION")
    print("█" * 80 + "\n")
    
    # Load trained model
    pipeline = load_trained_model('gbdt_lactation_risk_pipeline.pkl')
    
    if pipeline is None:
        print("✗ Cannot proceed without model file. Exiting...\n")
        exit(1)
    
    # Initialize SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(pipeline['model'])
    print("✓ SHAP explainer ready\n")
    
    # =========================================================================
    # EXAMPLE PREDICTION: Amoxicillin (Low Risk - L1)
    # =========================================================================
    
    print("\n" + "█" * 80)
    print("EXAMPLE 1: Amoxicillin (β-lactam antibiotic)")
    print("█" * 80 + "\n")
    
    amoxicillin_smiles = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
    
    predict_and_explain(
        smiles=amoxicillin_smiles,
        drug_name="Amoxicillin",
        pipeline=pipeline,
        explainer=explainer,
        show_descriptors=False,
        save_figures=False
    )
    
    # =========================================================================
    # USER CUSTOMIZATION SECTION
    # =========================================================================
    # 
    # TO PREDICT YOUR OWN DRUG:
    # 1. Replace the SMILES string below with your target molecule
    # 2. Update the drug_name
    # 3. Run this script
    #
    # Example SMILES codes:
    # - Aspirin:     CC(=O)Oc1ccccc1C(=O)O
    # - Ibuprofen:   CC(C)Cc1ccc(cc1)C(C)C(=O)O  
    # - Caffeine:    CN1C=NC2=C1C(=O)N(C(=O)N2C)C
    # - Warfarin:    CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O
    # - Metformin:   CN(C)C(=N)NC(=N)N
    #
    # =========================================================================
    
    print("\n" + "█" * 80)
    print("CUSTOM PREDICTION (Modify the SMILES and drug name below)")
    print("█" * 80 + "\n")
    
    # =========================================================================
    # MODIFY THESE TWO LINES FOR YOUR DRUG:
    # =========================================================================
    
    custom_smiles = "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1"
    custom_drug_name = "Amiodarone"
    
    # =========================================================================
    # DO NOT MODIFY BELOW THIS LINE
    # =========================================================================
    
    predict_and_explain(
        smiles=custom_smiles,
        drug_name=custom_drug_name,
        pipeline=pipeline,
        explainer=explainer,
        show_descriptors=False,  # Set to True to display all 35 descriptors
        save_figures=False       # Set to True to save plots as PNG files
    )
    
    # =========================================================================
    # END OF SCRIPT
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)
    print("\nFor more information:")
    print("  Web Platform: https://lrcpredictor.streamlit.app/")
    print("  GitHub: https://github.com/Huangxiaojie2024/LRCpredictor")
    print("  Contact: huangxj46@alumni.sysu.edu.cn")
    print("=" * 80 + "\n")
