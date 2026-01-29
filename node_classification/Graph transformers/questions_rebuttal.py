import os
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon
import collections
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# ==============================================================================
# OUTPUT HELPER
# ==============================================================================

class ReportLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.content = []
        
    def log(self, message=""):
        print(message)
        self.content.append(message)
        
    def save(self):
        with open(self.filepath, 'w') as f:
            f.write("\n".join(self.content))
        print(f"\n[Saved report to: {self.filepath}]")

# ==============================================================================
# DATA LOADING & PARSING
# ==============================================================================

def parse_folder_name(folder_name):
    """
    Parses folder strings to extract parameters. 
    Handles both 'lr-0.01_hid-64' and 'dim_128_layers_3_loss_focal' formats.
    """
    # Standardize separators
    normalized = folder_name.replace('-', '_')
    parts = normalized.split('_')
    
    params = {'loss': 'Standard', 'metric': 'rocauc', 'config_id': folder_name}
    
    # 1. Selection Metric Extraction
    if 'metric' in parts:
        idx = parts.index('metric')
        if idx + 1 < len(parts):
            params['metric'] = parts[idx + 1]
            
    # 2. Loss Type Extraction
    if 'focal' in parts:
        params['loss'] = 'Focal'
    elif 'class_weight' in parts or 'weighted' in parts:
        params['loss'] = 'Weighted'

    # 3. Stable Config ID Creation
    # Exclude run-specific and metric-specific tokens to group hyperparameter sets
    exclude = ['metric', params['metric'], 'run', 'gamma']
    config_parts = [p for p in parts if p not in exclude and not p.replace('.','',1).isdigit()]
    params['config_id'] = "_".join(config_parts)
    
    return params

def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
    experiments = []
    dataset_dir = os.path.join(base_dir, dataset)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return []

    for model_name in os.listdir(dataset_dir):
        if target_model and model_name.lower() != target_model.lower():
            continue

        model_path = os.path.join(dataset_dir, model_name)
        if not os.path.isdir(model_path): continue
        
        for folder_name in os.listdir(model_path):
            folder_path = os.path.join(model_path, folder_name)
            # Skip non-experiment or summary folders
            if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results', 'topk_percent']:
                continue
            
            if filter_str and filter_str not in folder_name:
                continue
            
            params = parse_folder_name(folder_name)
            topk_results_path = os.path.join(folder_path, 'topk_results')
            has_topk_folder = os.path.exists(topk_results_path)
            
            # Scan for runs (run_0, run_1... or run_1, run_2...)
            run_items = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
            for run_id in run_items:
                metrics_data = None
                
                # Priority 1: Check topk_results/run_X_metrics.json (from eval_ranking_transformers.py)
                if has_topk_folder:
                    json_name = f"{run_id}_metrics.json"
                    json_path = os.path.join(topk_results_path, json_name)
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f: metrics_data = json.load(f)
                        except: pass
                
                # Priority 2: Check inside the run folder itself
                if metrics_data is None:
                    json_path = os.path.join(folder_path, run_id, 'metrics.json')
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f: metrics_data = json.load(f)
                        except: pass
                
                if metrics_data:
                    entry = {
                        'model': model_name,
                        'config_id': params['config_id'],
                        'loss': params['loss'],
                        'selection_metric': params['metric'],
                        'run': run_id,
                        'data': metrics_data
                    }
                    experiments.append(entry)
                    
    return experiments

def get_k_metric(data, base_metric, k):
    # Try multiple naming conventions for Top-K
    keys_to_try = [
        f"{base_metric}_at_{k}_pct",
        f"{base_metric}_at_{float(k)}_pct",
        f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "NONE"
    ]
    for key in keys_to_try:
        if key in data: return data[key]
    return None

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def get_stat_str(values):
    if not values: return "-"
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

def analyze_part_a(experiments, logger):
    logger.log(f"\n## Part A: Best Model Performance (Per GNN/Transformer)\n")
    
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for exp in experiments:
        meta_key = (exp['model'], exp['loss'], exp['selection_metric'])
        grouped[meta_key][exp['config_id']].append(exp)
    
    sorted_meta_keys = sorted(grouped.keys())

    for metric_type in METRICS_OF_INTEREST:
        logger.log(f"### Ranking Table: {metric_type.upper()} Performance")
        k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
        header = f"| Model | Loss | Selection | Val Metric | Test ROC | Test PR | {k_headers} |"
        separator = "| :--- | :--- | :--- | :--- | :--- | :--- | " + " :--- |" * len(K_VALUES)
        
        logger.log(header)
        logger.log(separator)
        
        for meta_key in sorted_meta_keys:
            model, loss, sel_metric = meta_key
            configs = grouped[meta_key]
            
            # Find the best config based on Average Validation Metric
            best_config_id = None
            best_avg_val = -float('inf')
            
            for config_id, runs in configs.items():
                val_scores = [r['data'].get('val_metric', r['data'].get('best_validation_metric', 0)) for r in runs]
                avg_val = np.mean(val_scores)
                if avg_val > best_avg_val:
                    best_avg_val = avg_val
                    best_config_id = config_id
            
            best_runs = configs[best_config_id]
            
            # Map flexible keys for ROC/PR
            vals_val = [r['data'].get('val_metric', r['data'].get('best_validation_metric', 0)) for r in best_runs]
            vals_roc = [r['data'].get('test_roc_auc', r['data'].get('auc_roc', 0)) for r in best_runs]
            vals_pr  = [r['data'].get('test_pr_auc', r['data'].get('auc_pr', 0)) for r in best_runs]
            
            k_cols = []
            for k in K_VALUES:
                k_vals = [v for v in [get_k_metric(r['data'], metric_type, k) for r in best_runs] if v is not None]
                k_cols.append(get_stat_str(k_vals))
            
            logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {get_stat_str(vals_val)} | {get_stat_str(vals_roc)} | {get_stat_str(vals_pr)} | {' | '.join(k_cols)} |")
        logger.log("\n")

def perform_paired_analysis(experiments, logger, filter_loss=None):
    paired_data = collections.defaultdict(dict)
    for exp in experiments:
        if filter_loss and exp['loss'] != filter_loss: continue
        uid = (exp['model'], exp['config_id'], exp['run'])
        paired_data[uid][exp['selection_metric']] = exp
        
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    pair_count = 0
    
    for uid, metrics in paired_data.items():
        if 'prauc' in metrics and 'rocauc' in metrics:
            if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            pair_count += 1
            for m in METRICS_OF_INTEREST:
                for k in K_VALUES:
                    v_pr = get_k_metric(metrics['prauc']['data'], m, k)
                    v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
                    if v_pr is not None and v_roc is not None:
                        results[m][k].append((v_pr, v_roc))

    logger.log(f"### Paired Comparison: PR-AUC vs ROC-AUC ({filter_loss if filter_loss else 'All Loss Types'})")
    logger.log(f"*(Based on {pair_count} paired runs)*\n")
    logger.log("| Metric | Top-K | Gain (%) | Win Rate | P-Value |")
    logger.log("| :--- | :--- | :--- | :--- | :--- |")
    
    for m in METRICS_OF_INTEREST:
        first = True
        for k in K_VALUES:
            pairs = results[m][k]
            if not pairs: continue
            arr_pr = np.array([p[0] for p in pairs])
            arr_roc = np.array([p[1] for p in pairs])
            
            # Calculate % Gain using median ratio
            ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
            gain = (np.median(ratios) - 1) * 100
            win_rate = np.sum(arr_pr > arr_roc) / len(pairs) * 100
            
            try:
                p_val = wilcoxon(arr_pr, arr_roc)[1] if not np.all(arr_pr == arr_roc) else 1.0
            except: p_val = 1.0
            
            label = f"**{m.upper()}**" if first else ""
            p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            gain_str = f"{'+' if gain > 0 else ''}{gain:.2f}%"
            
            logger.log(f"| {label} | {k}% | {gain_str} | {win_rate:.1f}% | {p_str} |")
            first = False
        logger.log("| --- | --- | --- | --- | --- |")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="questions")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()
    
    # Path setup
    out_dir = os.path.join(args.base_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"report_{args.model if args.model else 'all'}.md")
    
    logger = ReportLogger(out_path)
    logger.log(f"# Graph Transformer & GNN Performance Report: {args.dataset.upper()}")
    
    exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
    if not exps:
        print("No metrics found. Ensure your 'topk_results' or 'run_X' folders contain 'metrics.json'.")
        return
    
    logger.log(f"Detected {len(exps)} valid runs.")
    
    analyze_part_a(exps, logger)
    logger.log(f"\n## Part B: Selection Metric Comparison")
    perform_paired_analysis(exps, logger)
    
    logger.save()

if __name__ == "__main__":
    main()