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
        """Load molecules from SDF file"""
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
                    'PUBCHEM_COMPOUND_CID': props.get('PUBCHEM_COMPOUND_CID', f'MOL_{idx}')
                }
                # Add other properties (such as labels)
                for key, value in props.items():
                    if key not in row:
                        row[key] = value
                data.append(row)
            else:
                print(f"Warning: Molecule at index {idx} could not be parsed")
        
        self.df = pd.DataFrame(data)
        print(f"✅ Successfully loaded {len(self.df)} molecules")
        print(f"📋 Available property columns: {[col for col in self.df.columns if col not in ['mol', 'SMILES', 'Index']]}\n")
        return self.df
    
    def calculate_rdkit_descriptors(self):
        """
        Calculate RDKit descriptors (200+ 1D/2D descriptors)
        
        Returns:
            DataFrame containing RDKit descriptors
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
                        'SMILES': row['SMILES']
                    }
                    result.update(dict(zip(descriptor_names, desc_values)))
                    results.append(result)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:  # Only show first 3 errors
                        print(f"  Warning: CID {cid} calculation failed - {str(e)}")
        
        df_descriptors = pd.DataFrame(results)
        output_file = os.path.join(self.output_dir, "01_RDKit_descriptors.csv")
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
            DataFrame containing Mordred descriptors
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
                        'SMILES': row['SMILES']
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
        
        output_file = os.path.join(self.output_dir, "02_Mordred_descriptors.csv")
        df_mordred.to_csv(output_file, index=False)
        
        print(f"✅ Successfully calculated: {len(results)}/{len(self.df)} molecules")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Features: {len(df_mordred.columns)-2} descriptors (invalid columns removed)")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} molecules")
        print()
        return df_mordred
    
    def calculate_maccs_keys(self):
        """
        Calculate MACCS keys fingerprints (166 bits)
        
        Returns:
            DataFrame containing MACCS keys
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
                        'SMILES': row['SMILES']
                    }
                    result.update({f'MACCS_{i}': int(fp_array[i]) for i in range(len(fp_array))})
                    results.append(result)
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        print(f"  Warning: CID {cid} calculation failed - {str(e)}")
        
        df_maccs = pd.DataFrame(results)
        output_file = os.path.join(self.output_dir, "03_MACCS_keys.csv")
        df_maccs.to_csv(output_file, index=False)
        
        print(f"✅ Successfully calculated: {len(results)}/{len(self.df)} molecules")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Features: 166 bits fingerprint")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} molecules")
        print()
        return df_maccs
    
    def generate_summary_report(self):
        """Generate summary report of calculated descriptors"""
        print("\n" + "="*70)
        print("📊 Descriptor Calculation Summary Report")
        print("="*70)
        
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        
        summary_data = []
        total_features = 0
        
        for csv_file in sorted(csv_files):
            filepath = os.path.join(self.output_dir, csv_file)
            df_temp = pd.read_csv(filepath, nrows=1)
            n_features = len(df_temp.columns) - 2  # Exclude CID and SMILES columns
            total_features += n_features
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            summary_data.append({
                'No.': csv_file.split('_')[0],
                'Filename': csv_file,
                'Features': n_features,
                'Size(MB)': f"{file_size:.2f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        print("="*70)
        print(f"Total: {len(csv_files)} feature files, {total_features} features")
        print("="*70)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "00_SUMMARY.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Molecular Descriptor Calculation Summary Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Input file: {self.sdf_file}\n")
            f.write(f"Total molecules: {len(self.df)}\n")
            f.write(f"Generated files: {len(csv_files)}\n")
            f.write(f"Total features: {total_features}\n\n")
            f.write(df_summary.to_string(index=False))
            f.write("\n\n")
            f.write("Descriptor Type Descriptions:\n")
            f.write("1. RDKit Descriptors - 200+ physicochemical property descriptors\n")
            f.write("2. Mordred Descriptors - 1800+ advanced molecular descriptors\n")
            f.write("3. MACCS Keys - 166-bit structural fingerprints\n")
        
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
        
        # Generate summary report
        self.generate_summary_report()
        
        print("✅ All descriptor calculations completed!")
        print(f"📁 Output directory: {self.output_dir}/")
        

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
    
    calculator = MolecularDescriptorCalculator(sdf_file)
    calculator.run_all()

if __name__ == "__main__":
    main()
