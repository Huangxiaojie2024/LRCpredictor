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
import matplotlib
matplotlib.use('Agg')  # 重要：使用非交互式后端

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


# ============ SHAP Force Plot 绘制函数（根据你的本地代码改编）============
def plot_shap_force_single_sample(
    X_test_std,
    X_test_original,
    best_model,
    drug_name
):
    """
    绘制单个测试样本的SHAP Force Plot（f(x)显示为预测概率）
    
    参数:
    - X_test_std: 标准化后的测试集 (DataFrame) - 单个样本
    - X_test_original: 原始测试集 (DataFrame) - 单个样本
    - best_model: 训练好的模型
    - drug_name: 药物名称
    
    返回:
    - fig: matplotlib figure对象
    """
    
    try:
        # 1. 计算SHAP值（概率空间）
        # 使用当前样本作为背景数据（因为只有一个样本）
        background_data = X_test_std.copy()
        
        explainer = shap.TreeExplainer(
            best_model, 
            data=background_data,
            feature_perturbation="interventional",
            model_output="probability"
        )
        
        shap_values_proba = explainer.shap_values(X_test_std)
        
        # 2. 处理SHAP值维度（二分类取正类）
        if isinstance(shap_values_proba, list):
            if len(shap_values_proba) == 2:
                shap_values_array = shap_values_proba[1]  # 二分类正类
            else:
                shap_values_array = np.mean(shap_values_proba, axis=0)
        else:
            shap_values_array = shap_values_proba
        
        # 3. 获取基准值（expected_value）- 概率空间的基准值
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) > 1:
                base_value = explainer.expected_value[1]  # 二分类取正类
            else:
                base_value = explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
        
        # 4. 获取样本数据（索引0，因为只有一个样本）
        sample_idx = 0
        sample_shap = shap_values_array[sample_idx, :]
        sample_original = X_test_original.iloc[sample_idx, :]
        
        # 5. 计算预测概率
        pred_proba = best_model.predict_proba(X_test_std)[0, 1]  # 正类概率
        
        # 6. 将原始特征值保留2位小数
        sample_original_rounded = np.round(sample_original.values, 2)
        
        # 7. 创建matplotlib图形 - 关键修改点
        fig = plt.figure(figsize=(20, 3), facecolor='white')
        ax = fig.add_subplot(111)
        
        # 8. 使用shap.force_plot绘制（matplotlib模式）
        shap.force_plot(
            base_value=base_value,
            shap_values=sample_shap,
            features=sample_original_rounded,
            feature_names=X_test_original.columns.tolist(),
            matplotlib=True,
            show=False,
            figsize=(20, 3)
        )
        
        # 9. 添加标题
        plt.title(
            f'SHAP Force Plot - {drug_name}\n'
            f'Base Value: {base_value:.4f} → Predicted Probability: {pred_proba:.4f}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP plot: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


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
    4. View SHAP force plot for model interpretability
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
        show_shap = st.checkbox("🔍 Show SHAP Force Plot (Model Explainability)", value=True)
        
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
                            
                            # SHAP Force Plot Visualization
                            if show_shap:
                                st.markdown("---")
                                st.markdown("### 🔍 SHAP Force Plot Analysis")
                                st.info("""
                                **Understanding the SHAP Force Plot:**
                                - **Base value**: Average prediction probability across all samples
                                - **Red features**: Push prediction towards HIGH risk (positive SHAP values)
                                - **Blue features**: Push prediction towards LOW risk (negative SHAP values)
                                - **Feature width**: Indicates the magnitude of impact on the prediction
                                """)
                                
                                with st.spinner("Generating SHAP force plot..."):
                                    # 获取模型
                                    model = pipeline['model']
                                    
                                    # 调用SHAP绘图函数
                                    shap_fig = plot_shap_force_single_sample(
                                        X_test_std=descriptor_std_df,
                                        X_test_original=descriptor_df,
                                        best_model=model,
                                        drug_name=drug_name
                                    )
                                    
                                    if shap_fig is not None:
                                        # 显示图形
                                        st.pyplot(shap_fig, use_container_width=True)
                                        # 关闭图形以释放内存
                                        plt.close(shap_fig)
                                    else:
                                        st.warning("⚠️ Unable to generate SHAP plot. Please check your data.")
                            
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
                            drug_name_batch = str(row['Drug Name'])
                            smiles = str(row['SMILES'])
                            
                            # Update progress
                            progress = (idx + 1) / len(batch_df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {idx + 1}/{len(batch_df)}: {drug_name_batch}")
                            
                            try:
                                # Calculate descriptors
                                descriptor_df_batch = calculate_all_descriptors(smiles, drug_name_batch)
                                
                                if descriptor_df_batch is not None:
                                    # Predict
                                    result, _ = predict_risk(descriptor_df_batch, pipeline)
                                    if result is not None:
                                        all_results.append(result)
                                    else:
                                        failed_drugs.append(drug_name_batch)
                                else:
                                    failed_drugs.append(drug_name_batch)
                            
                            except Exception as e:
                                failed_drugs.append(f"{drug_name_batch} (Error: {str(e)})")
                        
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
