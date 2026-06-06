import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# Part 1: Weight Coefficient Settings (Strategy Configuration)
# ==============================================================================

# 1. Base Weight
BASE_WEIGHT = 1.0

# 2. Biological Weight Coefficient (W_BIO)
# -----------------------------------------------------
# Bio Raw Score: 3 (three-way intersection), 2 (two-way intersection), 1 (union)
W_BIO = 1.0


# ==============================================================================
# Part 2: Data Loading and Preprocessing
# ==============================================================================

def normalize_name(name):
    """Normalize gene name: remove spaces, convert to uppercase"""
    if pd.isna(name): return ""
    return str(name).strip().upper()

def load_bio_data(filepath):
    """Load biological CSV file and extract sets"""
    print(f">>> Loading biological data: {filepath}")
    
    try:
        # Try to read, handle possible encoding issues (Chinese column names often use gb18030 or utf-8-sig)
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='gb18030')
            
        # Clean column name spaces
        df.columns = [c.strip() for c in df.columns]
        
        # Define key column names (must match your CSV headers)
        col_union = 'kegg∪go∪PPI' # Corresponds to 1 point
        col_inter2 = 'Union_of_pairwise_intersections'           # Corresponds to 2 points
        col_inter3 = 'kegg∩go∩PPI'            # Corresponds to 3 points
        
        # Check if columns exist
        for col in [col_union, col_inter2, col_inter3]:
            if col not in df.columns:
                print(f"!!! Error: Column '{col}' not found in {filepath}")
                print(f"    Current file columns: {list(df.columns)}")
                sys.exit(1)

        # Extract sets (remove null values)
        # Note: These columns in CSV are independent, not necessarily aligned, so need to get non-null values separately
        set_1 = set(df[col_union].dropna().apply(normalize_name))
        set_2 = set(df[col_inter2].dropna().apply(normalize_name))
        set_3 = set(df[col_inter3].dropna().apply(normalize_name))
        
        # Remove empty strings
        set_1.discard('')
        set_2.discard('')
        set_3.discard('')
        
        print(f"    - Union genes (1 point): {len(set_1)} genes")
        print(f"    - Two-way intersection (2 points): {len(set_2)} genes")
        print(f"    - Three-way intersection (3 points): {len(set_3)} genes")
        
        return set_1, set_2, set_3
        
    except FileNotFoundError:
        print(f"!!! Error: File not found {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"!!! Unknown error occurred when reading biological data: {e}")
        sys.exit(1)


# ==============================================================================
# Part 3: Calculation Logic
# ==============================================================================

def calculate_weight(gene, bio_sets):
    """Calculate final weight for a single gene"""
    g_norm = normalize_name(gene)
    bio_s1, bio_s2, bio_s3 = bio_sets
    
    # Calculate Bioinfo score (prioritize high scores)
    s_bio_raw = 0
    if g_norm in bio_s3: 
        s_bio_raw = 3
    elif g_norm in bio_s2: 
        s_bio_raw = 2
    elif g_norm in bio_s1: 
        s_bio_raw = 1
        
    s_bio_final = s_bio_raw * W_BIO
    
    # Summarize
    final_weight = BASE_WEIGHT + s_bio_final
    
    # Detail string
    breakdown = (
        f"Base:{BASE_WEIGHT} + "
        f"Bio:{s_bio_final:.1f}(Raw:{s_bio_raw})"
    )
    
    return final_weight, breakdown

# ==============================================================================
# Part 4: Main Program
# ==============================================================================

def main():
    # File path configuration
    file_bio = 'KEGG_GO_PPI.csv'
    file_input = 'gene.csv'                                     # Your target gene list
    file_output = 'final_external_data_weights.csv'
    
    print("-" * 60)
    print("Starting weighted analysis (based on external data sources)")
    print("-" * 60)
    
    # 1. Load data
    bio_sets = load_bio_data(file_bio)
    
    # 2. Read target gene list
    print(f">>> Reading target gene list: {file_input}")
    if not os.path.exists(file_input):
        # If not exists, generate a sample file for user testing
        print("    Note: Input file not found, generating sample gene.csv")
        sample_df = pd.DataFrame({'Gene': ['NFKBIA', 'OAS3', 'JAK1', 'UNKNOWN_GENE']})
        sample_df.to_csv(file_input, index=False)
        
    df_input = pd.read_csv(file_input)
    # Find gene column
    target_col = [c for c in df_input.columns if 'gene' in c.lower()]
    target_col = target_col[0] if target_col else df_input.columns[0]
    
    # 3. Calculate weights
    results = []
    for gene in df_input[target_col]:
        weight, detail = calculate_weight(gene, bio_sets)
        results.append({
            'Gene': gene,
            'Final_Weight': weight,
            'Calculation_Details': detail
        })
        
    # 4. Sort and save
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='Final_Weight', ascending=False)
    
    df_res.to_csv(file_output, index=False)
    
    print("-" * 60)
    print("Calculation completed! Top 5 gene weight preview:")
    print(df_res.head().to_string(index=False))
    print("-" * 60)
    print(f">>> Results saved to: {os.path.abspath(file_output)}")

if __name__ == '__main__':
    main()