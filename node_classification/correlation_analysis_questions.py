import os
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(model_path):
    """
    Loads data for a specific model by merging global metrics from 
    config/summary/summary.json and ranking metrics from config/topk_results/summary.json.
    """
    records = []
    
    if not os.path.exists(model_path):
        print(f"Error: Path {model_path} does not exist.")
        return None

    # Each directory under model_path is a unique hyperparameter configuration
    for config_name in os.listdir(model_path):
        config_dir = os.path.join(model_path, config_name)
        if not os.path.isdir(config_dir):
            continue
            
        # Define paths to the two specific summary files for this configuration
        roc_pr_file = os.path.join(config_dir, 'summary', 'summary.json')
        ndcg_file = os.path.join(config_dir, 'topk_results', 'summary.json')
        
        record = {'config': config_name}
        data_found = False
        
        # 1. Load ROC-AUC and PR-AUC from the 'summary' folder
        if os.path.exists(roc_pr_file):
            try:
                with open(roc_pr_file, 'r') as f:
                    data = json.load(f)
                    record['ROC-AUC'] = data.get('auc_roc_mean', data.get('auc_roc', 0))
                    record['PR-AUC'] = data.get('auc_pr_mean', data.get('auc_pr', 0))
                    data_found = True
            except Exception as e:
                print(f"Error reading ROC/PR for {config_name}: {e}")

        # 2. Load nDCG metrics from the 'topk_results' folder
        if os.path.exists(ndcg_file):
            try:
                with open(ndcg_file, 'r') as f:
                    data = json.load(f)
                    record['nDCG@0.5%'] = data.get('ndcg_at_0.5_pct_mean', 0)
                    record['nDCG@1%'] = data.get('ndcg_at_1_pct_mean', 0)
                    record['nDCG@2%'] = data.get('ndcg_at_2_pct_mean', 0)
                    record['nDCG@3%'] = data.get('ndcg_at_3_pct_mean', 0)
                    record['nDCG@4%'] = data.get('ndcg_at_4_pct_mean', 0)
                    record['nDCG@5%'] = data.get('ndcg_at_5_pct_mean', 0)
                    record['nDCG@10%'] = data.get('ndcg_at_10_pct_mean', 0)
                    data_found = True
            except Exception as e:
                print(f"Error reading nDCG for {config_name}: {e}")
                
        if data_found:
            records.append(record)
            
    return pd.DataFrame(records)

def plot_correlation(df, model_name, save_path, method='kendall'):
    """Calculates and plots the correlation matrix with p-value markers."""
    metrics = ['ROC-AUC', 'PR-AUC', 'nDCG@0.5%', 'nDCG@1%', 'nDCG@2%', 
               'nDCG@3%', 'nDCG@4%', 'nDCG@5%', 'nDCG@10%']
    
    df_filtered = df[[m for m in metrics if m in df.columns]]
    actual_metrics = df_filtered.columns.tolist()
    n = len(actual_metrics)
    
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    func = kendalltau if method == 'kendall' else spearmanr
    label = "Kendall Tau" if method == 'kendall' else "Spearman Rho"

    for i in range(n):
        for j in range(n):
            corr, p_val = func(df_filtered[actual_metrics[i]], df_filtered[actual_metrics[j]], nan_policy='omit')
            corr_matrix[i, j] = corr
            p_matrix[i, j] = p_val

    plt.figure(figsize=(12, 10))
    df_corr = pd.DataFrame(corr_matrix, index=actual_metrics, columns=actual_metrics)
    
    annot_labels = df_corr.copy().astype(str)
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            p = p_matrix[i, j]
            marker = ""
            if p < 0.01: marker = "**"
            elif p < 0.05: marker = "*"
            annot_labels.iloc[i, j] = f"{val:.2f}{marker}"

    sns.heatmap(df_corr, annot=annot_labels, fmt="", cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title(f"{label} Correlation: {model_name.upper()}\n(*p<0.05, **p<0.01)")
    plt.tight_layout()
    
    # Save the file in the provided save_path
    output_file = os.path.join(save_path, f"{method}_correlation_{model_name}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {label} matrix to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gcn)")
    parser.add_argument("--base_dir", type=str, default="results_weighted")
    parser.add_argument("--dataset", type=str, default="questions")
    args = parser.parse_args()

    # Construct the path to the model folder
    model_folder_path = os.path.join(args.base_dir, args.dataset, args.model)

    df = load_data(model_folder_path)

    # Assuming 'df' is the dataframe for GAT
    print("GAT Metric Stats:")
    print(df[['ROC-AUC', 'PR-AUC']].describe())
    
    # Check for constant values
    print("\nUnique ROC-AUC values:", df['ROC-AUC'].unique())
    print("Unique PR-AUC values:", df['PR-AUC'].unique())

    if df is not None and len(df) > 1:
        # Pass model_folder_path as the saving destination
        plot_correlation(df, args.model, model_folder_path, method='kendall')
        plot_correlation(df, args.model, model_folder_path, method='spearman')
    else:
        print(f"Insufficient data configurations found for model: {args.model}")

if __name__ == "__main__":
    main()


