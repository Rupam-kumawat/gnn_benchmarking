import os
import json
import pandas as pd
import glob
import re

# Configuration
RESULTS_DIR = "results_weighted/ogbg-molmuv"
K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 110, 10))# Percentages used in your MolMUV analysis

def parse_param_string(param_str):
    """Parses folder names to extract hyperparameters."""
    params = {}
    
    # Extract Target Metric
    if "metric-rocauc" in param_str:
        params['target_metric'] = 'rocauc'
    elif "metric-prauc" in param_str:
        params['target_metric'] = 'prauc'
    else:
        params['target_metric'] = 'unknown'

    # Extract Loss Type
    if "focal" in param_str:
        params['loss_type'] = 'focal'
    elif "class_weight" in param_str:
        params['loss_type'] = 'standard'
    else:
        params['loss_type'] = 'unknown'

    # Regex for numeric hyperparams
    hid_match = re.search(r'hid-(\d+)', param_str)
    do_match = re.search(r'do-([\d\.]+)', param_str)
    layers_match = re.search(r'layers-(\d+)', param_str)
    lr_match = re.search(r'lr-([\d\.]+)', param_str)
    
    params['hidden'] = int(hid_match.group(1)) if hid_match else 0
    params['dropout'] = float(do_match.group(1)) if do_match else 0.0
    params['layers'] = int(layers_match.group(1)) if layers_match else 0
    params['lr'] = float(lr_match.group(1)) if lr_match else 0.0
    
    return params

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory '{RESULTS_DIR}' not found.")
        return

    records = []
    print(f"Scanning {RESULTS_DIR} for global and ranking metrics...")
    
    # Look for the primary summary folder
    search_path = os.path.join(RESULTS_DIR, "*", "*", "summary", "summary.json")
    files = glob.glob(search_path)

    for filepath in files:
        try:
            path_parts = filepath.split(os.sep)
            param_string = path_parts[-3]
            model_name = path_parts[-4]
            config_dir = os.path.join(*path_parts[:-2]) # Path up to config folder
            
            config = parse_param_string(param_string)
            
            # 1. Load Global Metrics (ROC/PR)
            with open(filepath, 'r') as f:
                global_data = json.load(f)

            record = {
                'model': model_name,
                'layers': config['layers'],
                'hidden': config['hidden'],
                'dropout': config['dropout'],
                'lr': config['lr'],
                'loss_type': config['loss_type'],
                'target_metric': config['target_metric'],
                'test_rocauc': global_data.get('auc_roc_mean', 0),
                'test_prauc': global_data.get('auc_pr_mean', 0),
            }

            # 2. Load Top-K Metrics (nDCG/Precision) from topk_results folder
            topk_path = os.path.join(config_dir, "topk_results", "summary.json")
            if os.path.exists(topk_path):
                with open(topk_path, 'r') as f:
                    tk_data = json.load(f)
                    for k in K_VALUES:
                        record[f'ndcg@{k}%'] = tk_data.get(f'ndcg_at_{k}_pct_mean_mean', 
                                              tk_data.get(f'ndcg_at_{k}_pct_mean', 0))
                        record[f'prec@{k}%'] = tk_data.get(f'precision_at_{k}_pct_mean_mean', 
                                              tk_data.get(f'precision_at_{k}_pct_mean', 0))
            
            records.append(record)
        except Exception:
            pass

    df = pd.DataFrame(records)
    if df.empty: return

    target_models = ['gcn', 'sage', 'gat']
    categories = [("Standard Loss", "rocauc"), ("Standard Loss", "prauc"),
                  ("Focal Loss", "rocauc"), ("Focal Loss", "prauc")]

    for model in target_models:
        print("\n" + "="*95)
        print(f"  MODEL FAMILY: {model.upper()}")
        print("="*95)
        
        model_df = df[df['model'] == model]
        if model_df.empty: continue

        for loss_label, metric_target in categories:
            loss_key = 'standard' if 'Standard' in loss_label else 'focal'
            subset = model_df[(model_df['loss_type'] == loss_key) & (model_df['target_metric'] == metric_target)]

            if subset.empty: continue

            # Rank by the specific optimization target
            sort_col = 'test_rocauc' if metric_target == 'rocauc' else 'test_prauc'
            best_run = subset.sort_values(by=sort_col, ascending=False).iloc[0]

            print(f"\n  >>> Config: {loss_label} | Optimized for {metric_target.upper()}")
            print(f"      Params:   Layers={best_run['layers']}, Hidden={best_run['hidden']}, LR={best_run['lr']}")
            print(f"      GLOBAL:   Test ROC-AUC: {best_run['test_rocauc']:.4f} | Test PR-AUC: {best_run['test_prauc']:.4f}")
            
            # Print Top-K if available
            tk_strs = [f"nDCG@{k}%: {best_run.get(f'ndcg@{k}%', 0):.4f}" for k in K_VALUES]
            if any("ndcg" in k for k in best_run.keys()):
                print(f"      RANKING:  {' | '.join(tk_strs)}")

if __name__ == "__main__":
    main()





# import os
# import json
# import pandas as pd
# import glob
# import re

# # Configuration
# RESULTS_DIR = "results_weighted/ogbg-molmuv"

# def parse_param_string(param_str):
#     """
#     Parses the folder name string to extract hyperparameters.
#     """
#     params = {}
    
#     # Extract Metric (what the run was optimized for)
#     if "metric-rocauc" in param_str:
#         params['target_metric'] = 'rocauc'
#     elif "metric-prauc" in param_str:
#         params['target_metric'] = 'prauc'
#     else:
#         params['target_metric'] = 'unknown'

#     # Extract Loss Type
#     if "focal" in param_str:
#         params['loss_type'] = 'focal'
#     elif "class_weight" in param_str:
#         params['loss_type'] = 'standard'
#     else:
#         params['loss_type'] = 'unknown'

#     # Extract Hyperparams via Regex
#     hid_match = re.search(r'hid-(\d+)', param_str)
#     do_match = re.search(r'do-([\d\.]+)', param_str)
#     layers_match = re.search(r'layers-(\d+)', param_str)
#     lr_match = re.search(r'lr-([\d\.]+)', param_str)
    
#     params['hidden'] = int(hid_match.group(1)) if hid_match else 0
#     params['dropout'] = float(do_match.group(1)) if do_match else 0.0
#     params['layers'] = int(layers_match.group(1)) if layers_match else 0
#     params['lr'] = float(lr_match.group(1)) if lr_match else 0.0
    
#     return params

# def main():
#     if not os.path.exists(RESULTS_DIR):
#         print(f"Error: Directory '{RESULTS_DIR}' not found.")
#         return

#     records = []
#     print(f"Scanning {RESULTS_DIR}...")
    
#     # Look for summary.json files
#     search_path = os.path.join(RESULTS_DIR, "*", "*", "summary", "summary.json")
#     files = glob.glob(search_path)
#     print(f"Found {len(files)} completed runs. Parsing data...")

#     for filepath in files:
#         try:
#             path_parts = filepath.split(os.sep)
#             # Standard path: .../model_name/param_string/summary/summary.json
#             param_string = path_parts[-3]
#             model_name = path_parts[-4]

#             config = parse_param_string(param_string)
            
#             with open(filepath, 'r') as f:
#                 data = json.load(f)

#             record = {
#                 'model': model_name,
#                 'layers': config['layers'],
#                 'hidden': config['hidden'],
#                 'dropout': config['dropout'],
#                 'lr': config['lr'],
#                 'loss_type': config['loss_type'],
#                 'target_metric': config['target_metric'],
#                 # Metrics
#                 'test_rocauc': data.get('auc_roc_mean', 0),
#                 'test_prauc': data.get('auc_pr_mean', 0),
#                 'valid_score': data.get('best_valid_mean', 0)
#             }
#             records.append(record)
#         except Exception as e:
#             # print(f"Error parsing {filepath}: {e}")
#             pass

#     df = pd.DataFrame(records)
    
#     if df.empty:
#         print("No valid records found.")
#         return

#     # Define the specific models and categories we want
#     target_models = ['gcn', 'sage', 'gat']
#     categories = [
#         ("Standard Loss", "rocauc"),
#         ("Standard Loss", "prauc"),
#         ("Focal Loss",    "rocauc"),
#         ("Focal Loss",    "prauc"),
#     ]

#     # Iterate through each model
#     for model in target_models:
#         print("\n" + "="*80)
#         print(f"  MODEL FAMILY: {model.upper()}")
#         print("="*80)
        
#         # Filter dataframe for just this model
#         model_df = df[df['model'] == model]

#         if model_df.empty:
#             print(f"  No data found for {model}.")
#             continue

#         # Iterate through the 4 categories
#         for loss_label, metric_target in categories:
#             loss_key = 'standard' if 'Standard' in loss_label else 'focal'
            
#             # Filter: specific loss AND specific optimization target
#             subset = model_df[
#                 (model_df['loss_type'] == loss_key) & 
#                 (model_df['target_metric'] == metric_target)
#             ]

#             if subset.empty:
#                 print(f"  [Missing data for {loss_label} optimized for {metric_target}]")
#                 continue

#             # Sort by the metric we optimized for (the target_metric)
#             # If we optimized for PR-AUC, we want the run with the highest Test PR-AUC
#             sort_col = 'test_rocauc' if metric_target == 'rocauc' else 'test_prauc'
            
#             subset = subset.sort_values(by=sort_col, ascending=False)
#             best_run = subset.iloc[0]

#             print(f"\n  >>> Config: {loss_label} | Optimized for {metric_target.upper()}")
#             print(f"      Best Params:  Layers={best_run['layers']}, Hidden={best_run['hidden']}, Dropout={best_run['dropout']}, LR={best_run['lr']}")
#             print(f"      Test ROC-AUC: {best_run['test_rocauc']:.4f}")
#             print(f"      Test PR-AUC:  {best_run['test_prauc']:.4f}")

# if __name__ == "__main__":
#     main()
