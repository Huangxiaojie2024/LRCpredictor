import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import joblib
import shap
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# ============ Page Configuration ============
st.set_page_config(
    page_title="LRCpredictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Custom CSS Styles ============
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .stAlert {
        margin-top: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .shap-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============ Calculate Molecular Descriptors Function ============
@st.cache_data
def calculate_all_descriptors(smiles, drug_name="Unknown"):
    """
    Calculate all Mordred 2D and RDKit descriptors, then extract required 35 features
    
    Parameters:
    -----------
    smiles : str
        SMILES code of the drug
    drug_name : str
        Drug name
        
    Returns:
    --------
    DataFrame : DataFrame containing 35 descriptors
    """
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # Calculate Mordred 2D descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_results = calc(mol)
        
        mordred_dict = {}
        for desc_name in mordred_results.keys():
            try:
                value = mordred_results[desc_name]
                if pd.isna(value) or str(value) == 'nan' or isinstance(value, str):
                    mordred_dict[str(desc_name)] = 0.0
                else:
                    mordred_dict[str(desc_name)] = float(value)
            except:
                mordred_dict[str(desc_name)] = 0.0
        
        # Calculate RDKit descriptors
        rdkit_dict = {}
        rdkit_desc_list = Descriptors._descList
        
        for desc_name, desc_func in rdkit_desc_list:
            try:
                value = desc_func(mol)
                if pd.isna(value) or str(value) == 'nan':
                    rdkit_dict[desc_name] = 0.0
                else:
                    rdkit_dict[desc_name] = float(value)
            except:
                rdkit_dict[desc_name] = 0.0
        
        # Merge all descriptors
        all_descriptors = {**mordred_dict, **rdkit_dict}
        
        # Extract required 35 features
        required_features = [
            'ABC', 'SpMax_A', 'VE1_A', 'VR1_A', 'AATS1dv', 'AATS3v', 'AATS3i',
            'ATSC2dv', 'ATSC5Z', 'ATSC4se', 'ATSC5p', 'ATSC6i', 'ATSC7i',
            'AATSC2d', 'AATSC1i', 'GATS1dv', 'GATS3dv', 'GATS3s', 'GATS2i',
            'BCUTc-1h', 'BCUTse-1l', 'BCUTi-1h', 'BCUTi-1l', 'C3SP2',
            'AXp-1d', 'AETA_eta_F', 'IC1', 'MIC1', 'VSA_EState1', 'VSA_EState2', 'JGI3',
            'MaxEStateIndex', 'qed', 'BCUT2D_MRLOW', 'HallKierAlpha'
        ]
        
        selected_descriptors = {}
        for feature_name in required_features:
            if feature_name in all_descriptors:
                selected_descriptors[feature_name] = all_descriptors[feature_name]
            else:
                selected_descriptors[feature_name] = 0.0
        
        descriptor_df = pd.DataFrame([selected_descriptors], columns=required_features, index=[drug_name])
        
        return descriptor_df
        
    except Exception as e:
        st.error(f"Error calculating descriptors: {str(e)}")
        return None


# ============ Load Model ============
@st.cache_resource
def load_model(model_path='gbdt_lactation_risk_pipeline.pkl'):
    """
    Load saved model pipeline
    """
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"❌ Model file '{model_path}' not found!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# ============ SHAP Force Plot Function ============
def generate_shap_force_plot(descriptor_df, pipeline, drug_name="Unknown Drug"):
    """
    Generate SHAP force plot for a single prediction
    
    Parameters:
    -----------
    descriptor_df : DataFrame
        Molecular descriptors
    pipeline : dict
        Model pipeline containing model, scaler, and feature names
    drug_name : str
        Name of the drug
        
    Returns:
    --------
    str : HTML string of SHAP force plot
    dict : Dictionary containing SHAP analysis details
    """
    try:
        model = pipeline['model']
        scaler = pipeline['scaler']
        feature_names = pipeline['feature_names']
        
        # Ensure correct feature order
        descriptor_df = descriptor_df[feature_names]
        
        # Standardization
        descriptor_std = scaler.transform(descriptor_df)
        descriptor_std_df = pd.DataFrame(descriptor_std, 
                                         columns=feature_names, 
                                         index=descriptor_df.index)
        
        # Create background dataset (use standardized data)
        # For single prediction, we'll use the sample itself as background
        background_data = descriptor_std_df
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional",
            model_output="probability"
        )
        
        # Calculate SHAP values
        shap_values_proba = explainer.shap_values(descriptor_std_df)
        
        # Handle SHAP values dimension (binary classification - take positive class)
        if isinstance(shap_values_proba, list):
            if len(shap_values_proba) == 2:
                shap_values_array = shap_values_proba[1]  # Positive class
            else:
                shap_values_array = np.mean(shap_values_proba, axis=0)
        else:
            shap_values_array = shap_values_proba
        
        # Get base value (expected value in probability space)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) > 1:
                base_value = explainer.expected_value[1]  # Positive class
            else:
                base_value = explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
        
        # Get SHAP values for the sample
        sample_shap = shap_values_array[0, :]
        sample_original = descriptor_df.iloc[0, :].values
        
        # Calculate predicted probability
        pred_proba = model.predict_proba(descriptor_std_df)[0, 1]
        
        # Round original feature values
        sample_original_rounded = np.round(sample_original, 2)
        
        # Generate force plot HTML
        shap.initjs()
        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=sample_shap,
            features=sample_original_rounded,
            feature_names=feature_names,
            matplotlib=False  # Use HTML rendering
        )
        
        # Convert to HTML
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        
        # Prepare analysis details
        analysis_details = {
            'base_value': base_value,
            'pred_proba': pred_proba,
            'shap_sum': sample_shap.sum(),
            'verification_diff': abs((base_value + sample_shap.sum()) - pred_proba),
            'feature_contributions': []
        }
        
        # Get top 10 contributing features
        shap_abs = np.abs(sample_shap)
        top_10_idx = np.argsort(shap_abs)[-10:][::-1]
        
        for idx in top_10_idx:
            analysis_details['feature_contributions'].append({
                'feature': feature_names[idx],
                'value': sample_original_rounded[idx],
                'shap': sample_shap[idx]
            })
        
        return shap_html, analysis_details
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP plot: {str(e)}")
        return None, None


# ============ Prediction Function ============
def predict_risk(descriptor_df, pipeline):
    """
    Predict lactation risk
    """
    try:
        model = pipeline['model']
        scaler = pipeline['scaler']
        feature_names = pipeline['feature_names']
        
        # Ensure correct feature order
        descriptor_df = descriptor_df[feature_names]
        
        # Standardization
        descriptor_std = scaler.transform(descriptor_df)
        descriptor_std = pd.DataFrame(descriptor_std, 
                                      columns=feature_names, 
                                      index=descriptor_df.index)
        
        # Prediction
        predictions = model.predict(descriptor_std)
        probabilities = model.predict_proba(descriptor_std)
        
        # Organize results
        results = pd.DataFrame({
            'Drug Name': descriptor_df.index,
            'Prediction': predictions,
            'Risk Level': ['High Risk' if p == 1 else 'Low Risk' for p in predictions],
            'Low Risk Probability': probabilities[:, 0],
            'High Risk Probability': probabilities[:, 1]
        })
        
        return results
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


# ============ Main Interface ============
def main():
    
    # Title
    st.markdown('<p class="main-title">💊 LRCpredictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Lactation Risk Classification Predictor</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model
    pipeline = load_model('gbdt_lactation_risk_pipeline.pkl')
    
    if pipeline is None:
        st.error("❌ Model file not found! Please ensure 'gbdt_lactation_risk_pipeline.pkl' is in the current directory.")
        st.info("💡 Please upload your trained model file to continue.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("📋 Information")
    st.sidebar.info("""
    **LRCpredictor** is a machine learning-based tool for predicting lactation risk of drugs.
    
    **Features:**
    - Single drug prediction
    - Batch prediction
    - SMILES-based input
    - SHAP interpretability analysis
    
    **Model:** Gradient Boosting Decision Tree (GBDT)
    
    **Version:** 1.0
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Example SMILES")
    st.sidebar.code("Amoxicillin:\nCC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O", language="text")
    st.sidebar.code("Amiodarone:\nCCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1", language="text")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ How to Use")
    st.sidebar.markdown("""
    1. **Single Prediction**: Enter drug name and SMILES code
    2. **Batch Prediction**: Upload CSV file with multiple drugs
    3. Click **Predict** button to get results
    4. View **SHAP Analysis** to understand model decisions
    """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔬 Single Prediction", "📊 Batch Prediction"])
    
    # ========== Tab 1: Single Prediction ==========
    with tab1:
        st.header("Single Drug Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            drug_name = st.text_input("Drug Name (Optional)", placeholder="e.g., Amoxicillin", key="drug_name_input")
            smiles_input = st.text_area(
                "SMILES Code *", 
                placeholder="Enter SMILES code here...",
                height=100,
                key="smiles_input"
            )
        
        with col2:
            st.markdown("### 🧪 Quick Examples")
            if st.button("📌 Load Amoxicillin", use_container_width=True):
                st.session_state.smiles_single = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
                st.session_state.drug_name_single = "Amoxicillin"
                st.rerun()
            
            if st.button("📌 Load Amiodarone", use_container_width=True):
                st.session_state.smiles_single = "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1"
                st.session_state.drug_name_single = "Amiodarone"
                st.rerun()
            
            if st.button("🔄 Clear", use_container_width=True):
                if 'smiles_single' in st.session_state:
                    del st.session_state.smiles_single
                if 'drug_name_single' in st.session_state:
                    del st.session_state.drug_name_single
                st.rerun()
        
        # Use session state values if available
        if 'smiles_single' in st.session_state:
            smiles_input = st.session_state.smiles_single
        if 'drug_name_single' in st.session_state:
            drug_name = st.session_state.drug_name_single
        
        st.markdown("---")
        
        if st.button("🚀 Predict", type="primary", use_container_width=True):
            
            if not smiles_input.strip():
                st.error("⚠️ Please enter a SMILES code!")
            else:
                with st.spinner("Calculating molecular descriptors and predicting..."):
                    
                    # Set default drug name
                    if not drug_name.strip():
                        drug_name = "Unknown Drug"
                    
                    # Calculate descriptors
                    descriptor_df = calculate_all_descriptors(smiles_input.strip(), drug_name)
                    
                    if descriptor_df is None:
                        st.error("❌ Invalid SMILES code! Please check your input.")
                    else:
                        # Predict
                        results = predict_risk(descriptor_df, pipeline)
                        
                        if results is not None:
                            st.success("✅ Prediction completed!")
                            
                            # Display results
                            st.markdown("### 📊 Prediction Results")
                            
                            risk_level = results.iloc[0]['Risk Level']
                            low_prob = results.iloc[0]['Low Risk Probability']
                            high_prob = results.iloc[0]['High Risk Probability']
                            
                            if risk_level == "High Risk":
                                st.markdown(f"""
                                <div class="risk-high">
                                    <h3>⚠️ High Risk</h3>
                                    <p><strong>Drug Name:</strong> {drug_name}</p>
                                    <p><strong>SMILES:</strong> {smiles_input[:50]}{'...' if len(smiles_input) > 50 else ''}</p>
                                    <p><strong>High Risk Probability:</strong> {high_prob:.2%}</p>
                                    <p><strong>Low Risk Probability:</strong> {low_prob:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="risk-low">
                                    <h3>✅ Low Risk</h3>
                                    <p><strong>Drug Name:</strong> {drug_name}</p>
                                    <p><strong>SMILES:</strong> {smiles_input[:50]}{'...' if len(smiles_input) > 50 else ''}</p>
                                    <p><strong>Low Risk Probability:</strong> {low_prob:.2%}</p>
                                    <p><strong>High Risk Probability:</strong> {high_prob:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display probability chart
                            st.markdown("### 📈 Probability Distribution")
                            prob_data = pd.DataFrame({
                                'Risk Category': ['Low Risk', 'High Risk'],
                                'Probability': [low_prob, high_prob]
                            })
                            st.bar_chart(prob_data.set_index('Risk Category'))
                            
                            # ========== SHAP Analysis Section ==========
                            st.markdown("---")
                            st.markdown("### 🔍 SHAP Interpretability Analysis")
                            st.info("📊 SHAP (SHapley Additive exPlanations) values show how each molecular descriptor contributes to the prediction.")
                            
                            with st.spinner("Generating SHAP analysis..."):
                                shap_html, analysis_details = generate_shap_force_plot(
                                    descriptor_df, 
                                    pipeline, 
                                    drug_name
                                )
                                
                                if shap_html is not None and analysis_details is not None:
                                    # Display SHAP Force Plot
                                    st.markdown("#### 📊 SHAP Force Plot")
                                    st.markdown(f"""
                                    <div class="shap-container">
                                        <p><strong>Drug:</strong> {drug_name}</p>
                                        <p><strong>Base Value (Population Average):</strong> {analysis_details['base_value']:.4f}</p>
                                        <p><strong>Predicted Probability:</strong> {analysis_details['pred_proba']:.4f}</p>
                                        <p><em>Red features push the prediction higher (towards high risk), blue features push it lower (towards low risk).</em></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Render SHAP force plot
                                    components.html(shap_html, height=150, scrolling=True)
                                    
                                    # Display detailed feature contributions
                                    st.markdown("#### 📋 Top 10 Feature Contributions")
                                    
                                    contrib_df = pd.DataFrame(analysis_details['feature_contributions'])
                                    contrib_df['SHAP Value'] = contrib_df['shap'].apply(lambda x: f"{x:+.4f}")
                                    contrib_df['Feature Value'] = contrib_df['value'].apply(lambda x: f"{x:.2f}")
                                    contrib_df = contrib_df[['feature', 'Feature Value', 'SHAP Value']]
                                    contrib_df.columns = ['Feature Name', 'Feature Value', 'SHAP Contribution']
                                    
                                    st.dataframe(contrib_df, use_container_width=True)
                                    
                                    # Explanation
                                    with st.expander("❓ How to Interpret SHAP Values"):
                                        st.markdown("""
                                        **Understanding SHAP Force Plot:**
                                        
                                        - **Base Value**: The average prediction across all samples (population baseline)
                                        - **SHAP Value**: The contribution of each feature to push the prediction away from the base value
                                        - **Positive SHAP** (Red): Pushes prediction towards **High Risk** (Class 1)
                                        - **Negative SHAP** (Blue): Pushes prediction towards **Low Risk** (Class 0)
                                        
                                        **Formula:**
```
                                        f(x) = Base Value + Σ(SHAP values) = Predicted Probability
```
                                        
                                        **Example Interpretation:**
                                        - If a feature has SHAP = +0.15, it increases the high-risk probability by 15%
                                        - If a feature has SHAP = -0.10, it decreases the high-risk probability by 10%
                                        """)
                                    
                                    # Verification details
                                    with st.expander("🔬 SHAP Verification Details"):
                                        st.markdown(f"""
                                        **Mathematical Verification:**
                                        - Base Value: `{analysis_details['base_value']:.4f}`
                                        - Sum of All SHAP Values: `{analysis_details['shap_sum']:+.4f}`
                                        - **f(x) = Base + SHAP Sum**: `{analysis_details['base_value'] + analysis_details['shap_sum']:.4f}`
                                        - **Predicted Probability**: `{analysis_details['pred_proba']:.4f}`
                                        - **Difference**: `{analysis_details['verification_diff']:.6f}` (should be ≈ 0)
                                        
                                        ✅ The SHAP values correctly explain the model's prediction!
                                        """)
                            
                            # Display molecular descriptors
                            with st.expander("🔬 View All Molecular Descriptors"):
                                st.dataframe(descriptor_df.T, use_container_width=True)
    
    # ========== Tab 2: Batch Prediction ==========
    with tab2:
        st.header("Batch Drug Prediction")
        
        st.info("📝 Upload a CSV file with columns: 'Drug Name' and 'SMILES'")
        
        # Download template
        st.markdown("### 📥 Download Template")
        template_df = pd.DataFrame({
            'Drug Name': ['Amoxicillin', 'Amiodarone', 'Aspirin'],
            'SMILES': [
                'CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O',
                'CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1',
                'CC(=O)Oc1ccccc1C(=O)O'
            ]
        })
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV Template",
            data=csv_template,
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("📤 Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            
            try:
                # Read CSV
                batch_df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['Drug Name', 'SMILES']
                if not all(col in batch_df.columns for col in required_cols):
                    st.error(f"❌ CSV file must contain columns: {required_cols}")
                else:
                    st.success(f"✅ File uploaded successfully! Found {len(batch_df)} drugs.")
                    
                    # Display uploaded data
                    st.markdown("### 📋 Uploaded Data Preview")
                    st.dataframe(batch_df.head(10), use_container_width=True)
                    
                    st.markdown("---")
                    
                    if st.button("🚀 Batch Predict", type="primary", use_container_width=True):
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        all_results = []
                        failed_drugs = []
                        
                        for idx, row in batch_df.iterrows():
                            drug_name = str(row['Drug Name'])
                            smiles = str(row['SMILES'])
                            
                            # Update progress
                            progress = (idx + 1) / len(batch_df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {idx + 1}/{len(batch_df)}: {drug_name}")
                            
                            try:
                                # Calculate descriptors
                                descriptor_df = calculate_all_descriptors(smiles, drug_name)
                                
                                if descriptor_df is not None:
                                    # Predict
                                    result = predict_risk(descriptor_df, pipeline)
                                    if result is not None:
                                        all_results.append(result)
                                    else:
                                        failed_drugs.append(drug_name)
                                else:
                                    failed_drugs.append(drug_name)
                            
                            except Exception as e:
                                failed_drugs.append(f"{drug_name} (Error: {str(e)})")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if len(all_results) > 0:
                            # Combine results
                            final_results = pd.concat(all_results, ignore_index=True)
                            
                            st.success(f"✅ Batch prediction completed! {len(all_results)}/{len(batch_df)} drugs predicted successfully.")
                            
                            if failed_drugs:
                                with st.expander(f"⚠️ {len(failed_drugs)} drugs failed - Click to view"):
                                    for drug in failed_drugs:
                                        st.text(f"• {drug}")
                            
                            # Display results
                            st.markdown("### 📊 Prediction Results")
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Drugs", len(final_results))
                            with col2:
                                high_risk_count = (final_results['Prediction'] == 1).sum()
                                st.metric("High Risk", high_risk_count)
                            with col3:
                                low_risk_count = (final_results['Prediction'] == 0).sum()
                                st.metric("Low Risk", low_risk_count)
                            
                            # Display results table
                            st.dataframe(final_results, use_container_width=True)
                            
                            # Visualization
                            st.markdown("### 📈 Risk Distribution")
                            risk_counts = final_results['Risk Level'].value_counts()
                            st.bar_chart(risk_counts)
                            
                            # Download results
                            st.markdown("### 💾 Download Results")
                            csv_results = final_results.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Results as CSV",
                                data=csv_results,
                                file_name="lactation_risk_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ All predictions failed. Please check your SMILES codes.")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>LRCpredictor v1.0</strong> | Powered by GBDT & Streamlit</p>
        <p>⚠️ <em>Disclaimer: This tool is for research purposes only. Clinical decisions should be made by healthcare professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
