import os
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon
import collections
import warnings
import re

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Matches the K values used in your populate_molmuv_topk.py script
K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 110, 10))
METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

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
        print(f"\n[Report saved to: {self.filepath}]")

# ==============================================================================
# DATA LOADING
# ==============================================================================

def parse_folder_name(folder_name):
    """
    Parses folder string to extract parameters.
    Example: lr-0.0001_pool-mean_metric-rocauc_hid-64_...
    """
    params = {'loss': 'Standard', 'metric': 'rocauc'}
    
    # Identify Loss
    if 'class_weight' in folder_name or 'class-weight' in folder_name: 
        params['loss'] = 'Standard' 
    elif 'focal' in folder_name: 
        params['loss'] = 'Focal'
        
    # Identify Selection Metric
    if 'metric-rocauc' in folder_name:
        params['metric'] = 'rocauc'
    elif 'metric-prauc' in folder_name:
        params['metric'] = 'prauc'
    elif 'metric-ap' in folder_name:
        params['metric'] = 'prauc' # AP is often synonymous with PR-AUC here
        
    # Create a unique config ID by removing the metric part
    # This allows us to pair "Run A (Optimized for ROC)" with "Run A (Optimized for PR)"
    parts = folder_name.split('_')
    config_parts = sorted([p for p in parts if not p.startswith('metric')])
    params['config_id'] = "_".join(config_parts)
    
    return params

def load_all_results(base_dir, dataset, target_model=None):
    experiments = []
    dataset_dir = os.path.join(base_dir, dataset)
    if not os.path.exists(dataset_dir): 
        print(f"Dataset directory not found: {dataset_dir}")
        return []

    for model_name in os.listdir(dataset_dir):
        if target_model and model_name.lower() != target_model.lower(): continue
        
        model_path = os.path.join(dataset_dir, model_name)
        if not os.path.isdir(model_path): continue
        
        for folder_name in os.listdir(model_path):
            folder_path = os.path.join(model_path, folder_name)
            # Skip non-experiment folders
            if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results']:
                continue
            
            # Check for topk_results folder
            topk_path = os.path.join(folder_path, 'topk_results')
            if not os.path.exists(topk_path):
                continue

            params = parse_folder_name(folder_name)
            
            # Find all run folders
            run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_')]
            
            for run_id in run_dirs:
                # 1. Global Metrics (from training)
                orig_json = os.path.join(folder_path, run_id, 'metrics.json')
                # 2. Ranking Metrics (from post-hoc analysis)
                new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
                
                combined_data = {}
                
                # Load Global
                if os.path.exists(orig_json):
                    try:
                        with open(orig_json, 'r') as f:
                            combined_data.update(json.load(f))
                    except: pass
                
                # Load Ranking
                if os.path.exists(new_json):
                    try:
                        with open(new_json, 'r') as f:
                            combined_data.update(json.load(f))
                    except: pass
                
                # Only add if we have data
                if combined_data:
                    experiments.append({
                        'model': model_name,
                        'config_id': params['config_id'],
                        'loss': params['loss'],
                        'selection_metric': params['metric'],
                        'run': run_id,
                        'data': combined_data
                    })
    return experiments

def get_k_metric(data, base_metric, k):
    """
    Robustly fetches metric for k%. 
    Handles keys like 'ndcg_at_1_pct_mean' (from molmuv script) and 'ndcg_at_1_pct' (legacy).
    """
    # Keys created by populate_molmuv_topk.py usually have _mean suffix because they are averaged over tasks
    candidates = [
        f"{base_metric}_at_{k}_pct_mean",
        f"{base_metric}_at_{k}_pct",
        f"{base_metric}_at_{k}_mean"
    ]
    for key in candidates:
        if key in data:
            return data[key]
    return None

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_part_a(experiments, logger):
    logger.log(f"\n## Part A: Full Spectrum Performance Decay (Mean ± Std)\n")
    groups = collections.defaultdict(list)
    for exp in experiments:
        groups[(exp['model'], exp['loss'], exp['selection_metric'])].append(exp)
    
    for m_type in METRICS_OF_INTEREST:
        # check if this metric exists in at least one experiment
        if not any(get_k_metric(e['data'], m_type, K_VALUES[0]) is not None for e in experiments):
            continue

        logger.log(f"### Metric: {m_type.upper()}")
        header = "| Model | Loss | Selection | Val Score | ROC-AUC | PR-AUC | " + " | ".join([f"@{k}%" for k in K_VALUES]) + " |"
        sep = "| :--- " * (6 + len(K_VALUES)) + "|"
        logger.log(header); logger.log(sep)
        
        for key in sorted(groups.keys()):
            model, loss, sel = key
            c_groups = collections.defaultdict(list)
            for e in groups[key]: c_groups[e['config_id']].append(e)
            
            # Identify the "Best Config" based on the validation metric (Validation ROC or PR)
            # This simulates realistic model selection: we pick the hyperparameters that did best on val
            best_cid = max(c_groups.keys(), key=lambda c: np.mean([
                e['data'].get('best_validation_metric', e['data'].get('auc_roc', 0)) 
                for e in c_groups[c]
            ]))
            best_runs = c_groups[best_cid]
            
            def get_stat(key_name):
                vals = [e['data'].get(key_name, 0) for e in best_runs]
                # Filter out None/NaN
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if not vals: return "-"
                return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

            v_str = get_stat('best_validation_metric')
            r_str = get_stat('auc_roc')
            p_str = get_stat('auc_pr')
            
            k_strs = []
            for k in K_VALUES:
                vals = [get_k_metric(e['data'], m_type, k) for e in best_runs]
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                k_strs.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "0.000")
            
            logger.log(f"| {model} | {loss} | {sel.upper()} | {v_str} | {r_str} | {p_str} | {' | '.join(k_strs)} |")
        logger.log("\n")

def perform_paired_analysis(experiments, logger):
    # Pair experiments by (Model, ConfigID, RunID)
    # matching ROC-optimized vs PR-optimized versions of the SAME run
    paired = collections.defaultdict(dict)
    for exp in experiments:
        uid = (exp['model'], exp['config_id'], exp['run'])
        paired[uid][exp['selection_metric']] = exp
        
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    
    # We need both 'prauc' and 'rocauc' results for the same config to compare
    count = 0
    for uid, metrics in paired.items():
        if 'prauc' in metrics and 'rocauc' in metrics:
            count += 1
            for m in METRICS_OF_INTEREST:
                for k in K_VALUES:
                    v_pr = get_k_metric(metrics['prauc']['data'], m, k)
                    v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
                    
                    if v_pr is not None and v_roc is not None:
                        results[m][k].append((v_pr, v_roc))

    logger.log(f"## Part B: Selection Strategy Impact (PR vs ROC Selection)")
    logger.log(f"Found {count} paired runs for direct comparison.")
    
    for m in METRICS_OF_INTEREST:
        if not results[m]: continue
        logger.log(f"### {m.upper()} Comparison")
        logger.log("| Top-K | Avg Gain | Win Rate | P-Value |")
        logger.log("| :--- | :--- | :--- | :--- |")
        
        for k in K_VALUES:
            pairs = results[m][k]
            if not pairs: continue
            
            a_pr = np.array([p[0] for p in pairs])
            a_roc = np.array([p[1] for p in pairs])
            
            # Calculate Gain
            # Avoid div by zero
            gain = (np.mean((a_pr + 1e-9) / (a_roc + 1e-9)) - 1) * 100
            
            # Calculate Win Rate (ties count as 0.5 win)
            wins = np.sum(a_pr > a_roc)
            ties = np.sum(a_pr == a_roc)
            win_rate = (wins + 0.5 * ties) / len(pairs) * 100
            
            # Wilcoxon Test
            try:
                if np.all(a_pr == a_roc):
                    p_val = 1.0
                else:
                    p_val = wilcoxon(a_pr, a_roc)[1]
            except:
                p_val = 1.0
            
            p_mark = "**" if p_val < 0.05 else ""
            logger.log(f"| {k}% | {gain:+.1f}% | {win_rate:.1f}% | {p_val:.2e}{p_mark} |")
        logger.log("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results_weighted")
    parser.add_argument("--dataset", type=str, default="ogbg-molmuv")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., gcn)")
    args = parser.parse_args()
    
    exps = load_all_results(args.base_dir, args.dataset, args.model)
    
    if not exps: 
        print("No experiments found. Ensure you have run populate_molmuv_topk.py first!")
        return
    
    out_path = os.path.join(args.base_dir, args.dataset, "molmuv_rebuttal_report.md")
    
    logger = ReportLogger(out_path)
    logger.log(f"# MolMUV Rebuttal Analysis")
    logger.log(f"Dataset: {args.dataset}\n")
    
    analyze_part_a(exps, logger)
    perform_paired_analysis(exps, logger)
    logger.save()

if __name__ == "__main__":
    main()