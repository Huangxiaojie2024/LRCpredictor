import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MolecularDescriptorCalculator:
    """
    Molecular Descriptor Calculator for Machine Learning Applications
    
    Calculates three types of molecular descriptors:
    1. RDKit Descriptors (200+ physicochemical properties)
    2. Mordred Descriptors (1800+ advanced molecular descriptors)
    3. MACCS Keys (166-bit structural fingerprints)
    
    Preserves original SDF properties: PUBCHEM_COMPOUND_CID, Toxicity, Data Set
    """
    
    def __init__(self, sdf_file, output_dir="molecular_descriptors"):
        """
        Initialize the calculator
        
        Args:
            sdf_file: Path to input SDF file
            output_dir: Directory for output files
        """
        self.sdf_file = sdf_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.df = None
        
    def load_sdf(self):
        """Load molecules from SDF file and preserve all properties"""
        print("Loading SDF file...")
        suppl = Chem.SDMolSupplier(self.sdf_file)
        
        data = []
        for idx, mol in enumerate(suppl):
            if mol is not None:
                props = mol.GetPropsAsDict()
                row = {
                    'Index': idx,
                    'mol': mol,
                    'SMILES': Chem.MolToSmiles(mol),
                    'PUBCHEM_COMPOUND_CID': props.get('PUBCHEM_COMPOUND_CID', f'MOL_{idx}'),
                    'Toxicity': props.get('Toxicity', None),
                    'Data Set': props.get('Data Set', None)
                }
                # Add any other properties
                for key, value in props.items():
                    if key not in row:
                        row[key] = value
                data.append(row)
            else:
                print(f"Warning: Molecule at index {idx} could not be parsed")
        
        self.df = pd.DataFrame(data)
        print(f"✅ Successfully loaded {len(self.df)} molecules")
        print(f"📋 Available property columns: {[col for col in self.df.columns if col not in ['mol', 'SMILES', 'Index']]}\n")
        
        # Display dataset distribution if available
        if 'Data Set' in self.df.columns and self.df['Data Set'].notna().any():
            print("Dataset distribution:")
            print(self.df['Data Set'].value_counts())
            print()
        
        if 'Toxicity' in self.df.columns and self.df['Toxicity'].notna().any():
            print("Toxicity label distribution:")
            print(self.df['Toxicity'].value_counts())
            print()
        
        return self.df
    
    def calculate_rdkit_descriptors(self):
        """
        Calculate RDKit descriptors (200+ 1D/2D descriptors)
        
        Returns:
            DataFrame containing RDKit descriptors with preserved labels
        """
        print("="*70)
        print("📊 1. Calculating RDKit Descriptors (200+ descriptors)")
        print("="*70)
        
        # Get all available descriptors
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        
        results = []
        failed_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="RDKit Descriptors"):
            mol = row['mol']
            cid = row['PUBCHEM_COMPOUND_CID']
            
            if mol is not None:
                try:
                    desc_values = calculator.CalcDescriptors(mol)
                    result = {
                        'PUBCHEM_COMPOUND_CID': cid,
                        'SMILES': row['SMILES'],
                        'Toxicity': row.get('Toxicity'),
                        'Data Set': row.get('Data Set')
                    }
                    result.update(dict(zip(descriptor_names, desc_values)))
                    results.append(result)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:  # Only show first 3 errors
                        print(f"  Warning: CID {cid} calculation failed - {str(e)}")
        
        df_descriptors = pd.DataFrame(results)
        
        # Reorder columns: ID, SMILES, Toxicity, Data Set, then descriptors
        id_cols = ['PUBCHEM_COMPOUND_CID', 'SMILES', 'Toxicity', 'Data Set']
        desc_cols = [col for col in df_descriptors.columns if col not in id_cols]
        df_descriptors = df_descriptors[id_cols + desc_cols]
        
        output_file = os.path.join(self.output_dir, "RDKit.csv")
        df_descriptors.to_csv(output_file, index=False)
        
        print(f"✅ Successfully calculated: {len(results)}/{len(self.df)} molecules")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Features: {len(descriptor_names)} descriptors")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} molecules")
        print()
        return df_descriptors
    
    def calculate_mordred_descriptors(self):
        """
        Calculate Mordred descriptors (1800+ advanced descriptors)
        
        Returns:
            DataFrame containing Mordred descriptors with preserved labels
        """
        print("="*70)
        print("📊 2. Calculating Mordred Descriptors (1800+ descriptors)")
        print("="*70)
        print("⚠️  Note: Mordred calculation may take time, please be patient...")
        
        # Create Mordred calculator
        calc = Calculator(descriptors, ignore_3D=True)
        
        results = []
        failed_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Mordred Descriptors"):
            mol = row['mol']
            cid = row['PUBCHEM_COMPOUND_CID']
            
            if mol is not None:
                try:
                    # Calculate all descriptors
                    desc_values = calc(mol)
                    result = {
                        'PUBCHEM_COMPOUND_CID': cid,
                        'SMILES': row['SMILES'],
                        'Toxicity': row.get('Toxicity'),
                        'Data Set': row.get('Data Set')
                    }
                    
                    # Convert to dictionary
                    for desc_name, value in zip(calc.descriptors, desc_values):
                        # Handle potential error values
                        if isinstance(value, (int, float, np.number)):
                            if not (np.isnan(value) or np.isinf(value)):
                                result[str(desc_name)] = value
                            else:
                                result[str(desc_name)] = None
                        else:
                            result[str(desc_name)] = None
                    
                    results.append(result)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        print(f"  Warning: CID {cid} calculation failed - {str(e)}")
        
        df_mordred = pd.DataFrame(results)
        
        # Remove columns with all None values
        df_mordred = df_mordred.dropna(axis=1, how='all')
        
        # Reorder columns: ID, SMILES, Toxicity, Data Set, then descriptors
        id_cols = ['PUBCHEM_COMPOUND_CID', 'SMILES', 'Toxicity', 'Data Set']
        desc_cols = [col for col in df_mordred.columns if col not in id_cols]
        df_mordred = df_mordred[id_cols + desc_cols]
        
        output_file = os.path.join(self.output_dir, "Mordred.csv")
        df_mordred.to_csv(output_file, index=False)
        
        print(f"✅ Successfully calculated: {len(results)}/{len(self.df)} molecules")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Features: {len(df_mordred.columns)-4} descriptors (invalid columns removed)")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} molecules")
        print()
        return df_mordred
    
    def calculate_maccs_keys(self):
        """
        Calculate MACCS keys fingerprints (166 bits)
        
        Returns:
            DataFrame containing MACCS keys with preserved labels
        """
        print("="*70)
        print("🔑 3. Calculating MACCS Keys (166 bits)")
        print("="*70)
        
        results = []
        failed_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="MACCS Keys"):
            mol = row['mol']
            cid = row['PUBCHEM_COMPOUND_CID']
            
            if mol is not None:
                try:
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    fp_array = np.array(fp)
                    result = {
                        'PUBCHEM_COMPOUND_CID': cid,
                        'SMILES': row['SMILES'],
                        'Toxicity': row.get('Toxicity'),
                        'Data Set': row.get('Data Set')
                    }
                    result.update({f'MACCS_{i}': int(fp_array[i]) for i in range(len(fp_array))})
                    results.append(result)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        print(f"  Warning: CID {cid} calculation failed - {str(e)}")
        
        df_maccs = pd.DataFrame(results)
        
        # Reorder columns: ID, SMILES, Toxicity, Data Set, then MACCS bits
        id_cols = ['PUBCHEM_COMPOUND_CID', 'SMILES', 'Toxicity', 'Data Set']
        maccs_cols = [col for col in df_maccs.columns if col not in id_cols]
        df_maccs = df_maccs[id_cols + maccs_cols]
        
        output_file = os.path.join(self.output_dir, "MACCS.csv")
        df_maccs.to_csv(output_file, index=False)
        
        print(f"✅ Successfully calculated: {len(results)}/{len(self.df)} molecules")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Features: 166 bits fingerprint")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} molecules")
        print()
        return df_maccs
    
    def merge_descriptors(self):
        """
        Merge different descriptor types to create combined feature sets
        """
        print("="*70)
        print("🔗 4. Merging Descriptor Files")
        print("="*70)
        
        # Load all descriptor files
        rdkit_file = os.path.join(self.output_dir, "RDKit.csv")
        mordred_file = os.path.join(self.output_dir, "Mordred.csv")
        maccs_file = os.path.join(self.output_dir, "MACCS.csv")
        
        if not all(os.path.exists(f) for f in [rdkit_file, mordred_file, maccs_file]):
            print("⚠️  Not all descriptor files found. Skipping merge step.")
            return
        
        df_rdkit = pd.read_csv(rdkit_file)
        df_mordred = pd.read_csv(mordred_file)
        df_maccs = pd.read_csv(maccs_file)
        
        # Common columns for merging
        merge_cols = ['PUBCHEM_COMPOUND_CID', 'SMILES', 'Toxicity', 'Data Set']
        
        # Create combined datasets
        combinations = [
            ('RDKit+MACCS', df_rdkit, df_maccs),
            ('Mordred+MACCS', df_mordred, df_maccs),
            ('Mordred+RDKit', df_mordred, df_rdkit),
        ]
        
        for name, df1, df2 in combinations:
            # Get descriptor columns only (exclude merge columns)
            df1_desc = df1.drop(columns=[col for col in merge_cols if col in df1.columns and col != 'PUBCHEM_COMPOUND_CID'])
            df2_desc = df2.drop(columns=[col for col in merge_cols if col in df2.columns and col != 'PUBCHEM_COMPOUND_CID'])
            
            # Merge
            df_combined = pd.merge(df1_desc, df2_desc, on='PUBCHEM_COMPOUND_CID', how='inner')
            
            # Reorder columns
            id_cols_present = [col for col in merge_cols if col in df_combined.columns]
            desc_cols = [col for col in df_combined.columns if col not in merge_cols]
            df_combined = df_combined[id_cols_present + desc_cols]
            
            output_file = os.path.join(self.output_dir, f"{name}.csv")
            df_combined.to_csv(output_file, index=False)
            print(f"  ✅ Created {name}.csv ({len(desc_cols)} features)")
        
        # Create all three combined
        df_rdkit_desc = df_rdkit.drop(columns=[col for col in merge_cols if col in df_rdkit.columns and col != 'PUBCHEM_COMPOUND_CID'])
        df_mordred_desc = df_mordred.drop(columns=[col for col in merge_cols if col in df_mordred.columns and col != 'PUBCHEM_COMPOUND_CID'])
        df_maccs_desc = df_maccs.drop(columns=[col for col in merge_cols if col in df_maccs.columns and col != 'PUBCHEM_COMPOUND_CID'])
        
        df_all = pd.merge(df_rdkit_desc, df_mordred_desc, on='PUBCHEM_COMPOUND_CID', how='inner')
        df_all = pd.merge(df_all, df_maccs_desc, on='PUBCHEM_COMPOUND_CID', how='inner')
        
        id_cols_present = [col for col in merge_cols if col in df_all.columns]
        desc_cols = [col for col in df_all.columns if col not in merge_cols]
        df_all = df_all[id_cols_present + desc_cols]
        
        output_file = os.path.join(self.output_dir, "Mordred+RDKit+MACCS.csv")
        df_all.to_csv(output_file, index=False)
        print(f"  ✅ Created Mordred+RDKit+MACCS.csv ({len(desc_cols)} features)")
        print()
    
    def generate_summary_report(self):
        """Generate summary report of calculated descriptors"""
        print("\n" + "="*70)
        print("📊 Descriptor Calculation Summary Report")
        print("="*70)
        
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        
        summary_data = []
        
        for csv_file in sorted(csv_files):
            filepath = os.path.join(self.output_dir, csv_file)
            df_temp = pd.read_csv(filepath, nrows=1)
            # Exclude ID columns from feature count
            n_features = len(df_temp.columns) - 4  # Subtract CID, SMILES, Toxicity, Data Set
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            summary_data.append({
                'Filename': csv_file,
                'Features': n_features,
                'Size(MB)': f"{file_size:.2f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        print("="*70)
        print(f"Total: {len(csv_files)} feature files")
        print("="*70)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "00_SUMMARY.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Molecular Descriptor Calculation Summary Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Input file: {self.sdf_file}\n")
            f.write(f"Total molecules: {len(self.df)}\n")
            f.write(f"Generated files: {len(csv_files)}\n\n")
            
            if 'Data Set' in self.df.columns and self.df['Data Set'].notna().any():
                f.write("Dataset distribution:\n")
                f.write(str(self.df['Data Set'].value_counts()) + "\n\n")
            
            if 'Toxicity' in self.df.columns and self.df['Toxicity'].notna().any():
                f.write("Toxicity label distribution:\n")
                f.write(str(self.df['Toxicity'].value_counts()) + "\n\n")
            
            f.write(df_summary.to_string(index=False))
            f.write("\n\n")
            f.write("Descriptor Type Descriptions:\n")
            f.write("1. RDKit.csv - 200+ physicochemical property descriptors\n")
            f.write("2. Mordred.csv - 1800+ advanced molecular descriptors\n")
            f.write("3. MACCS.csv - 166-bit structural fingerprints\n")
            f.write("4. RDKit+MACCS.csv - Combined RDKit and MACCS features\n")
            f.write("5. Mordred+MACCS.csv - Combined Mordred and MACCS features\n")
            f.write("6. Mordred+RDKit.csv - Combined Mordred and RDKit features\n")
            f.write("7. Mordred+RDKit+MACCS.csv - All three descriptor types combined\n")
            f.write("\nNote: All files preserve original labels (Toxicity, Data Set)\n")
        
        print(f"\n✅ Summary report saved to: {summary_file}\n")
    
    def run_all(self):
        """Run all descriptor calculations"""
        print("\n" + "🧬 "*25)
        print("Molecular Descriptor Calculator for Machine Learning")
        print("🧬 "*25 + "\n")
        
        # Load data
        self.load_sdf()
        
        # Calculate all descriptors in order
        self.calculate_rdkit_descriptors()
        self.calculate_mordred_descriptors()
        self.calculate_maccs_keys()
        
        # Merge descriptor files
        self.merge_descriptors()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("✅ All descriptor calculations completed!")
        print(f"📁 Output directory: {self.output_dir}/")
        print("\n💡 Generated Files:")
        print("   • RDKit.csv - RDKit descriptors only")
        print("   • Mordred.csv - Mordred descriptors only")
        print("   • MACCS.csv - MACCS keys only")
        print("   • RDKit+MACCS.csv - Combined features")
        print("   • Mordred+MACCS.csv - Combined features")
        print("   • Mordred+RDKit.csv - Combined features")
        print("   • Mordred+RDKit+MACCS.csv - All features combined")
        print("\n⚠️  Note: All files preserve Toxicity and Data Set labels\n")

def main():
    """
    Main function to run the molecular descriptor calculator
    
    Usage:
        Place your SDF file in the current directory and update the filename below
    """
    sdf_file = "your_molecules.sdf"  # Update with your SDF filename
    
    if not os.path.exists(sdf_file):
        print(f"❌ Error: File '{sdf_file}' not found")
        print("Please ensure the SDF file is in the current directory")
        return
    
    calculator = MolecularDescriptorCalculator(sdf_file, output_dir="DataSet")
    calculator.run_all()

if __name__ == "__main__":
    main()
