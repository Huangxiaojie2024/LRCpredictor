import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import joblib
import warnings
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import base64
from io import BytesIO

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


# ============ SHAP Force Plot Generation Function ============
def generate_shap_force_plot(descriptor_df, descriptor_std_df, pipeline, drug_name, top_n=10):
    """
    Generate SHAP Force Plot for a single sample
    
    Parameters:
    -----------
    descriptor_df : DataFrame
        Original descriptor values
    descriptor_std_df : DataFrame
        Standardized descriptor values
    pipeline : dict
        Model pipeline containing model and scaler
    drug_name : str
        Name of the drug
    top_n : int
        Number of top features to highlight
        
    Returns:
    --------
    tuple : (matplotlib figure, shap_info dict)
    """
    
    try:
        model = pipeline['model']
        scaler = pipeline['scaler']
        
        # Create background data for SHAP
        background_data = descriptor_std_df.sample(n=min(50, len(descriptor_std_df)), random_state=42)
        
        # Initialize TreeExplainer with probability output
        explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional",
            model_output="probability"
        )
        
        # Calculate SHAP values
        shap_values_proba = explainer.shap_values(descriptor_std_df)
        
        # Handle different SHAP value formats
        if isinstance(shap_values_proba, list):
            if len(shap_values_proba) == 2:
                shap_values_array = shap_values_proba[1]  # Binary classification - positive class
            else:
                shap_values_array = np.mean(shap_values_proba, axis=0)
        else:
            shap_values_array = shap_values_proba
        
        # Get base value (expected value)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) > 1:
                base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
        
        # Get SHAP values for the sample
        sample_shap = shap_values_array[0, :]
        sample_original = descriptor_df.iloc[0, :]
        
        # Calculate prediction probability
        pred_proba = model.predict_proba(descriptor_std_df)[0, 1]
        
        # Get top N features by absolute SHAP value
        feature_importance = pd.DataFrame({
            'Feature': descriptor_df.columns,
            'SHAP_Value': sample_shap,
            'Abs_SHAP': np.abs(sample_shap),
            'Original_Value': sample_original.values
        }).sort_values('Abs_SHAP', ascending=False)
        
        top_features = feature_importance.head(top_n)
        
        # Create matplotlib force plot
        fig, ax = plt.subplots(figsize=(14, 3))
        
        # Round original values for display
        sample_original_rounded = np.round(sample_original.values, 2)
        
        # Create force plot using matplotlib
        shap.force_plot(
            base_value=base_value,
            shap_values=sample_shap,
            features=sample_original_rounded,
            feature_names=descriptor_df.columns.tolist(),
            matplotlib=True,
            show=False,
            figsize=(14, 3)
        )
        
        plt.title(
            f'SHAP Force Plot - {drug_name}\n'
            f'Base Value: {base_value:.4f} → Predicted Probability: {pred_proba:.4f}',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        plt.tight_layout()
        
        # Prepare SHAP info
        shap_info = {
            'base_value': base_value,
            'pred_proba': pred_proba,
            'top_features': top_features,
            'total_shap_sum': sample_shap.sum(),
            'verification': abs((base_value + sample_shap.sum()) - pred_proba)
        }
        
        return fig, shap_info
        
    except Exception as e:
        st.error(f"Error generating SHAP plot: {str(e)}")
        return None, None


# ============ Generate SHAP HTML for Interactive Display ============
def st_shap_force_plot(shap_values, base_value, features, feature_names):
    """
    Display SHAP force plot in Streamlit using HTML/JavaScript
    """
    try:
        shap.initjs()
        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values,
            features=features,
            feature_names=feature_names
        )
        
        # Save to HTML
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(shap_html, height=300)
        
    except Exception as e:
        st.warning(f"Interactive SHAP plot unavailable, showing matplotlib version instead. Error: {str(e)}")
        return False
    
    return True


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
        descriptor_std_df = pd.DataFrame(
            descriptor_std,
            columns=feature_names,
            index=descriptor_df.index
        )
        
        # Prediction
        predictions = model.predict(descriptor_std_df)
        probabilities = model.predict_proba(descriptor_std_df)
        
        # Organize results
        results = pd.DataFrame({
            'Drug Name': descriptor_df.index,
            'Prediction': predictions,
            'Risk Level': ['High Risk' if p == 1 else 'Low Risk' for p in predictions],
            'Low Risk Probability': probabilities[:, 0],
            'High Risk Probability': probabilities[:, 1]
        })
        
        return results, descriptor_std_df
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None


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
    - SHAP explainability
    
    **Model:** Gradient Boosting Decision Tree (GBDT)
    
    **Version:** 1.1
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
    4. View SHAP analysis for model interpretability
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
        
        # Add SHAP option
        show_shap = st.checkbox("🔍 Show SHAP Analysis (Model Explainability)", value=True)
        
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
                        results, descriptor_std_df = predict_risk(descriptor_df, pipeline)
                        
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
                            
                            # SHAP Analysis
                            if show_shap:
                                st.markdown("---")
                                st.markdown("### 🔍 SHAP Model Explainability Analysis")
                                st.info("SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction.")
                                
                                with st.spinner("Generating SHAP analysis..."):
                                    fig, shap_info = generate_shap_force_plot(
                                        descriptor_df,
                                        descriptor_std_df,
                                        pipeline,
                                        drug_name,
                                        top_n=10
                                    )
                                    
                                    if fig is not None and shap_info is not None:
                                        # Display Force Plot
                                        st.pyplot(fig)
                                        plt.close()
                                        
                                        # Display SHAP statistics
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Base Probability", f"{shap_info['base_value']:.4f}")
                                        with col2:
                                            st.metric("Predicted Probability", f"{shap_info['pred_proba']:.4f}")
                                        with col3:
                                            st.metric("Verification Error", f"{shap_info['verification']:.6f}")
                                        
                                        # Display top features contribution
                                        st.markdown("#### 🎯 Top 10 Feature Contributions")
                                        
                                        top_features_display = shap_info['top_features'][['Feature', 'Original_Value', 'SHAP_Value']].copy()
                                        top_features_display['Original_Value'] = top_features_display['Original_Value'].round(4)
                                        top_features_display['SHAP_Value'] = top_features_display['SHAP_Value'].round(4)
                                        top_features_display['Impact'] = top_features_display['SHAP_Value'].apply(
                                            lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Decreases Risk'
                                        )
                                        
                                        st.dataframe(
                                            top_features_display,
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                        
                                        # Interpretation guide
                                        with st.expander("📖 How to Interpret SHAP Values"):
                                            st.markdown("""
                                            **Understanding the SHAP Force Plot:**
                                            
                                            - **Base Value**: The average prediction probability across all samples
                                            - **Red bars**: Features pushing the prediction towards HIGH risk
                                            - **Blue bars**: Features pushing the prediction towards LOW risk
                                            - **Final Prediction**: Base value + sum of all SHAP values
                                            
                                            **SHAP Value Interpretation:**
                                            - Positive SHAP value (>0): Feature increases risk probability
                                            - Negative SHAP value (<0): Feature decreases risk probability
                                            - Larger absolute value: Stronger influence on prediction
                                            """)
                            
                            # Display molecular descriptors
                            with st.expander("🔬 View Molecular Descriptors"):
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
                                    result, _ = predict_risk(descriptor_df, pipeline)
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
        <p><strong>LRCpredictor v1.1</strong> | Powered by GBDT & Streamlit | Enhanced with SHAP Explainability</p>
        <p>⚠️ <em>Disclaimer: This tool is for research purposes only. Clinical decisions should be made by healthcare professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
