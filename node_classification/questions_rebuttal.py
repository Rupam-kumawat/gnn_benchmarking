import os
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon
import collections
import warnings
import sys

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
# DATA LOADING
# ==============================================================================

def parse_folder_name(folder_name):
    """Parses folder string to extract parameters."""
    parts = folder_name.split('_')
    params = {}
    params['loss'] = 'Standard'
    params['metric'] = 'rocauc'
    
    if 'class_weight' in parts: params['loss'] = 'Standard' 
    elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
    for part in parts:
        if part.startswith('metric'):
            raw_val = part.replace('metric', '')
            if raw_val.startswith('-'): raw_val = raw_val[1:]
            params['metric'] = raw_val
    
    config_parts = [p for p in parts if not p.startswith('metric')]
    config_parts.sort()
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
            if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
                continue
            
            if filter_str and filter_str not in folder_name:
                continue
            
            params = parse_folder_name(folder_name)
            topk_results_path = os.path.join(folder_path, 'topk_results')
            has_topk_folder = os.path.exists(topk_results_path)
            
            for item in os.listdir(folder_path):
                if not item.startswith('run_'): continue
                if not os.path.isdir(os.path.join(folder_path, item)): continue
                
                run_id = item 
                metrics_data = None
                
                if has_topk_folder:
                    json_name = f"{run_id}_metrics.json"
                    json_path = os.path.join(topk_results_path, json_name)
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f: metrics_data = json.load(f)
                        except: pass
                
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
    keys_to_try = [
        f"{base_metric}_at_{k}_pct",
        f"{base_metric}_at_{k:.1f}_pct",
        f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
    ]
    for key in keys_to_try:
        if key in data: return data[key]
    return None

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def get_stat_str(values):
    """Helper to format list of values into Mean +/- Std."""
    if not values:
        return "-"
    mean_val = np.mean(values)
    std_val = np.std(values)
    return f"{mean_val:.4f} ± {std_val:.4f}"

def analyze_part_a(experiments, logger):
    logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
    logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.")
    logger.log("Values are reported as **Mean ± StdDev** across runs/seeds.\n")
    
    # 1. Group experiments by (Model, Loss, Selection Metric) -> Config ID -> List of Runs
    # structure: grouped[meta_key][config_id] = [exp1, exp2...]
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for exp in experiments:
        if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
        meta_key = (exp['model'], exp['loss'], exp['selection_metric'])
        grouped[meta_key][exp['config_id']].append(exp)
    
    sorted_meta_keys = sorted(grouped.keys())

    for metric_type in METRICS_OF_INTEREST:
        logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
        k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
        header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
        sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
        separator = "".join(sep_cols)
        
        logger.log(header)
        logger.log(separator)
        
        for meta_key in sorted_meta_keys:
            model, loss, sel_metric = meta_key
            configs = grouped[meta_key]
            
            # 2. Find the best configuration based on Average Validation Metric
            best_config_id = None
            best_avg_val = -float('inf')
            
            for config_id, runs in configs.items():
                val_scores = [r['data'].get('best_validation_metric', 0) for r in runs]
                avg_val = np.mean(val_scores)
                if avg_val > best_avg_val:
                    best_avg_val = avg_val
                    best_config_id = config_id
            
            # 3. Aggregate metrics for the best configuration
            best_runs = configs[best_config_id]
            
            # Collect values for stats
            vals_val = [r['data'].get('best_validation_metric', 0) for r in best_runs]
            vals_roc = [r['data'].get('auc_roc', 0) for r in best_runs]
            vals_pr  = [r['data'].get('auc_pr', 0) for r in best_runs]
            
            k_cols = []
            for k in K_VALUES:
                k_vals = []
                for r in best_runs:
                    val = get_k_metric(r['data'], metric_type, k)
                    if val is not None: k_vals.append(val)
                k_cols.append(get_stat_str(k_vals))
            
            val_str = get_stat_str(vals_val)
            roc_str = get_stat_str(vals_roc)
            pr_str = get_stat_str(vals_pr)
            k_row_str = " | ".join(k_cols)
            
            logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val_str} | {roc_str} | {pr_str} | {k_row_str} |")
        
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

    logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
    logger.log(f"*(Based on {pair_count} comparisons)*\n")
    
    # Updated Header to include Mean Ratio and % Gain
    logger.log("| Metric | Top-K | Mean Ratio (Volatile) | Agg Ratio (Total) | Median Ratio (Typical) | % Gain (Robust) | Win Rate | P-Value |")
    logger.log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for m in METRICS_OF_INTEREST:
        first = True
        for k in K_VALUES:
            pairs = results[m][k]
            if not pairs: continue
            arr_pr = np.array([p[0] for p in pairs])
            arr_roc = np.array([p[1] for p in pairs])
            
            # 1. Mean Ratio (Sensitive to outliers)
            individual_ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
            mean_ratio = np.mean(individual_ratios)
            
            # 2. Aggregate Ratio (Robust to individual 0s)
            agg_ratio = np.sum(arr_pr) / (np.sum(arr_roc) + 1e-9)
            
            # 3. Median Ratio (Typical performance)
            median_ratio = np.median(individual_ratios)
            
            # 4. % Gain (Relative Improvement based on Median)
            pct_gain = (median_ratio - 1) * 100
            
            win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
            try: 
                diff = arr_pr - arr_roc
                p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
            except: p_val = 1.0
            
            # Formatting strings
            label = f"**{m.upper()}**" if first else ""
            p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            
            # Handle exploding Mean Ratio display
            if mean_ratio > 100: mean_str = f"{mean_ratio:.1e}x"
            else: mean_str = f"{mean_ratio:.3f}x"
                
            agg_str = f"**{agg_ratio:.3f}x**" if agg_ratio > 1.0 else f"{agg_ratio:.3f}x"
            med_str = f"**{median_ratio:.3f}x**" if median_ratio > 1.0 else f"{median_ratio:.3f}x"
            
            # Gain string with +/- sign
            gain_str = f"+{pct_gain:.1f}%" if pct_gain > 0 else f"{pct_gain:.1f}%"
            if pct_gain > 0: gain_str = f"**{gain_str}**"
            
            logger.log(f"| {label} | {k}% | {mean_str} | {agg_str} | {med_str} | {gain_str} | {win:.1f}% | {p_str} |")
            first = False
        logger.log("| | | | | | | | |")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results_weighted")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()
    
    output_filename = f"analysis_report_{args.dataset}"
    if args.model: output_filename += f"_{args.model}"
    if args.filter: output_filename += f"_{args.filter}"
    output_filename += ".md"
    
    out_path = os.path.join(args.base_dir, args.dataset, output_filename)
    logger = ReportLogger(out_path)
    
    header = f"# Analysis Report for {args.dataset}"
    if args.model: header += f" ({args.model})"
    if args.filter: header += f" [Filter: {args.filter}]"
    logger.log(header)
    
    exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
    if not exps: 
        print("No results found.")
        return
    
    logger.log(f"Loaded {len(exps)} experimental runs.")
    
    analyze_part_a(exps, logger)
    logger.log(f"\n## Part B: The 'Ratio Test'")
    perform_paired_analysis(exps, logger, None)
    perform_paired_analysis(exps, logger, 'Standard')
    perform_paired_analysis(exps, logger, 'Focal')
    
    logger.save()

if __name__ == "__main__":
    main()








# import os
# import json
# import argparse
# import numpy as np
# from scipy.stats import wilcoxon
# import collections
# import warnings
# import sys

# warnings.filterwarnings("ignore")

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================

# K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # ==============================================================================
# # OUTPUT HELPER
# # ==============================================================================

# class ReportLogger:
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.content = []
        
#     def log(self, message=""):
#         print(message)
#         self.content.append(message)
        
#     def save(self):
#         with open(self.filepath, 'w') as f:
#             f.write("\n".join(self.content))
#         print(f"\n[Saved report to: {self.filepath}]")

# # ==============================================================================
# # DATA LOADING
# # ==============================================================================

# def parse_folder_name(folder_name):
#     """Parses folder string to extract parameters."""
#     parts = folder_name.split('_')
#     params = {}
#     params['loss'] = 'Standard'
#     params['metric'] = 'rocauc'
    
#     if 'class_weight' in parts: params['loss'] = 'Standard' 
#     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
#     for part in parts:
#         if part.startswith('metric'):
#             raw_val = part.replace('metric', '')
#             if raw_val.startswith('-'): raw_val = raw_val[1:]
#             params['metric'] = raw_val
    
#     config_parts = [p for p in parts if not p.startswith('metric')]
#     config_parts.sort()
#     params['config_id'] = "_".join(config_parts)
#     return params

# def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
#     experiments = []
#     dataset_dir = os.path.join(base_dir, dataset)
    
#     if not os.path.exists(dataset_dir):
#         print(f"Error: Dataset directory not found: {dataset_dir}")
#         return []

#     for model_name in os.listdir(dataset_dir):
#         if target_model and model_name.lower() != target_model.lower():
#             continue

#         model_path = os.path.join(dataset_dir, model_name)
#         if not os.path.isdir(model_path): continue
        
#         for folder_name in os.listdir(model_path):
#             folder_path = os.path.join(model_path, folder_name)
#             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
#                 continue
            
#             if filter_str and filter_str not in folder_name:
#                 continue
            
#             params = parse_folder_name(folder_name)
#             topk_results_path = os.path.join(folder_path, 'topk_results')
#             has_topk_folder = os.path.exists(topk_results_path)
            
#             for item in os.listdir(folder_path):
#                 if not item.startswith('run_'): continue
#                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
#                 run_id = item 
#                 metrics_data = None
                
#                 if has_topk_folder:
#                     json_name = f"{run_id}_metrics.json"
#                     json_path = os.path.join(topk_results_path, json_name)
#                     if os.path.exists(json_path):
#                         try:
#                             with open(json_path, 'r') as f: metrics_data = json.load(f)
#                         except: pass
                
#                 if metrics_data is None:
#                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
#                     if os.path.exists(json_path):
#                         try:
#                             with open(json_path, 'r') as f: metrics_data = json.load(f)
#                         except: pass
                
#                 if metrics_data:
#                     entry = {
#                         'model': model_name,
#                         'config_id': params['config_id'],
#                         'loss': params['loss'],
#                         'selection_metric': params['metric'],
#                         'run': run_id,
#                         'data': metrics_data
#                     }
#                     experiments.append(entry)
                    
#     return experiments

# def get_k_metric(data, base_metric, k):
#     keys_to_try = [
#         f"{base_metric}_at_{k}_pct",
#         f"{base_metric}_at_{k:.1f}_pct",
#         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
#     ]
#     for key in keys_to_try:
#         if key in data: return data[key]
#     return None

# # ==============================================================================
# # ANALYSIS FUNCTIONS
# # ==============================================================================

# def analyze_part_a(experiments, logger):
#     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
#     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
#     groups = collections.defaultdict(list)
#     for exp in experiments:
#         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
#         key = (exp['model'], exp['loss'], exp['selection_metric'])
#         groups[key].append(exp)
    
#     sorted_keys = sorted(groups.keys())

#     for metric_type in METRICS_OF_INTEREST:
#         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
#         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
#         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
#         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
#         separator = "".join(sep_cols)
        
#         logger.log(header)
#         logger.log(separator)
        
#         for key in sorted_keys:
#             model, loss, sel_metric = key
#             exps = groups[key]
            
#             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
#             d = best_exp['data']
            
#             val = d.get('best_validation_metric', -1)
#             roc = d.get('auc_roc', 0)
#             pr = d.get('auc_pr', 0)
            
#             k_strings = []
#             for k in K_VALUES:
#                 val_k = get_k_metric(d, metric_type, k)
#                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
#                 k_strings.append(val_str)
            
#             k_row_str = " | ".join(k_strings)
            
#             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
#         logger.log("\n")

# def perform_paired_analysis(experiments, logger, filter_loss=None):
#     paired_data = collections.defaultdict(dict)
#     for exp in experiments:
#         if filter_loss and exp['loss'] != filter_loss: continue
#         uid = (exp['model'], exp['config_id'], exp['run'])
#         paired_data[uid][exp['selection_metric']] = exp
        
#     results = collections.defaultdict(lambda: collections.defaultdict(list))
#     pair_count = 0
    
#     for uid, metrics in paired_data.items():
#         if 'prauc' in metrics and 'rocauc' in metrics:
#             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
#             pair_count += 1
#             for m in METRICS_OF_INTEREST:
#                 for k in K_VALUES:
#                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
#                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
#                     if v_pr is not None and v_roc is not None:
#                         results[m][k].append((v_pr, v_roc))

#     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
#     logger.log(f"*(Based on {pair_count} comparisons)*\n")
    
#     # Updated Header to include Mean Ratio and % Gain
#     logger.log("| Metric | Top-K | Mean Ratio (Volatile) | Agg Ratio (Total) | Median Ratio (Typical) | % Gain (Robust) | Win Rate | P-Value |")
#     logger.log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
#     for m in METRICS_OF_INTEREST:
#         first = True
#         for k in K_VALUES:
#             pairs = results[m][k]
#             if not pairs: continue
#             arr_pr = np.array([p[0] for p in pairs])
#             arr_roc = np.array([p[1] for p in pairs])
            
#             # 1. Mean Ratio (Sensitive to outliers)
#             individual_ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
#             mean_ratio = np.mean(individual_ratios)
            
#             # 2. Aggregate Ratio (Robust to individual 0s)
#             agg_ratio = np.sum(arr_pr) / (np.sum(arr_roc) + 1e-9)
            
#             # 3. Median Ratio (Typical performance)
#             median_ratio = np.median(individual_ratios)
            
#             # 4. % Gain (Relative Improvement based on Median)
#             pct_gain = (median_ratio - 1) * 100
            
#             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
#             try: 
#                 diff = arr_pr - arr_roc
#                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
#             except: p_val = 1.0
            
#             # Formatting strings
#             label = f"**{m.upper()}**" if first else ""
#             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            
#             # Handle exploding Mean Ratio display
#             if mean_ratio > 100: mean_str = f"{mean_ratio:.1e}x"
#             else: mean_str = f"{mean_ratio:.3f}x"
                
#             agg_str = f"**{agg_ratio:.3f}x**" if agg_ratio > 1.0 else f"{agg_ratio:.3f}x"
#             med_str = f"**{median_ratio:.3f}x**" if median_ratio > 1.0 else f"{median_ratio:.3f}x"
            
#             # Gain string with +/- sign
#             gain_str = f"+{pct_gain:.1f}%" if pct_gain > 0 else f"{pct_gain:.1f}%"
#             if pct_gain > 0: gain_str = f"**{gain_str}**"
            
#             logger.log(f"| {label} | {k}% | {mean_str} | {agg_str} | {med_str} | {gain_str} | {win:.1f}% | {p_str} |")
#             first = False
#         logger.log("| | | | | | | | |")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_dir", type=str, default="results_weighted")
#     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
#     parser.add_argument("--model", type=str, default=None)
#     parser.add_argument("--filter", type=str, default=None)
#     args = parser.parse_args()
    
#     output_filename = f"analysis_report_{args.dataset}"
#     if args.model: output_filename += f"_{args.model}"
#     if args.filter: output_filename += f"_{args.filter}"
#     output_filename += ".md"
    
#     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
#     logger = ReportLogger(out_path)
    
#     header = f"# Analysis Report for {args.dataset}"
#     if args.model: header += f" ({args.model})"
#     if args.filter: header += f" [Filter: {args.filter}]"
#     logger.log(header)
    
#     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
#     if not exps: 
#         print("No results found.")
#         return
    
#     logger.log(f"Loaded {len(exps)} experimental runs.")
    
#     analyze_part_a(exps, logger)
#     logger.log(f"\n## Part B: The 'Ratio Test'")
#     perform_paired_analysis(exps, logger, None)
#     perform_paired_analysis(exps, logger, 'Standard')
#     perform_paired_analysis(exps, logger, 'Focal')
    
#     logger.save()

# if __name__ == "__main__":
#     main()







# # import os
# # import json
# # import argparse
# # import numpy as np
# # from scipy.stats import wilcoxon
# # import collections
# # import warnings
# # import sys

# # warnings.filterwarnings("ignore")

# # # ==============================================================================
# # # CONFIGURATION
# # # ==============================================================================

# # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # ==============================================================================
# # # OUTPUT HELPER
# # # ==============================================================================

# # class ReportLogger:
# #     def __init__(self, filepath):
# #         self.filepath = filepath
# #         self.content = []
        
# #     def log(self, message=""):
# #         print(message)
# #         self.content.append(message)
        
# #     def save(self):
# #         with open(self.filepath, 'w') as f:
# #             f.write("\n".join(self.content))
# #         print(f"\n[Saved report to: {self.filepath}]")

# # # ==============================================================================
# # # DATA LOADING
# # # ==============================================================================

# # def parse_folder_name(folder_name):
# #     parts = folder_name.split('_')
# #     params = {}
# #     params['loss'] = 'Standard'
# #     params['metric'] = 'rocauc'
    
# #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# #     for part in parts:
# #         if part.startswith('metric'):
# #             raw_val = part.replace('metric', '')
# #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# #             params['metric'] = raw_val
    
# #     config_parts = [p for p in parts if not p.startswith('metric')]
# #     config_parts.sort()
# #     params['config_id'] = "_".join(config_parts)
# #     return params

# # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# #     experiments = []
# #     dataset_dir = os.path.join(base_dir, dataset)
    
# #     if not os.path.exists(dataset_dir):
# #         print(f"Error: Dataset directory not found: {dataset_dir}")
# #         return []

# #     for model_name in os.listdir(dataset_dir):
# #         if target_model and model_name.lower() != target_model.lower():
# #             continue

# #         model_path = os.path.join(dataset_dir, model_name)
# #         if not os.path.isdir(model_path): continue
        
# #         for folder_name in os.listdir(model_path):
# #             folder_path = os.path.join(model_path, folder_name)
# #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# #                 continue
            
# #             if filter_str and filter_str not in folder_name:
# #                 continue
            
# #             params = parse_folder_name(folder_name)
# #             topk_results_path = os.path.join(folder_path, 'topk_results')
# #             has_topk_folder = os.path.exists(topk_results_path)
            
# #             for item in os.listdir(folder_path):
# #                 if not item.startswith('run_'): continue
# #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# #                 run_id = item 
# #                 metrics_data = None
                
# #                 if has_topk_folder:
# #                     json_name = f"{run_id}_metrics.json"
# #                     json_path = os.path.join(topk_results_path, json_name)
# #                     if os.path.exists(json_path):
# #                         try:
# #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# #                         except: pass
                
# #                 if metrics_data is None:
# #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# #                     if os.path.exists(json_path):
# #                         try:
# #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# #                         except: pass
                
# #                 if metrics_data:
# #                     entry = {
# #                         'model': model_name,
# #                         'config_id': params['config_id'],
# #                         'loss': params['loss'],
# #                         'selection_metric': params['metric'],
# #                         'run': run_id,
# #                         'data': metrics_data
# #                     }
# #                     experiments.append(entry)
                    
# #     return experiments

# # def get_k_metric(data, base_metric, k):
# #     keys_to_try = [
# #         f"{base_metric}_at_{k}_pct",
# #         f"{base_metric}_at_{k:.1f}_pct",
# #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# #     ]
# #     for key in keys_to_try:
# #         if key in data: return data[key]
# #     return None

# # # ==============================================================================
# # # ANALYSIS FUNCTIONS
# # # ==============================================================================

# # def analyze_part_a(experiments, logger):
# #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# #     groups = collections.defaultdict(list)
# #     for exp in experiments:
# #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# #         groups[key].append(exp)
    
# #     sorted_keys = sorted(groups.keys())

# #     for metric_type in METRICS_OF_INTEREST:
# #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# #         separator = "".join(sep_cols)
        
# #         logger.log(header)
# #         logger.log(separator)
        
# #         for key in sorted_keys:
# #             model, loss, sel_metric = key
# #             exps = groups[key]
            
# #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# #             d = best_exp['data']
            
# #             val = d.get('best_validation_metric', -1)
# #             roc = d.get('auc_roc', 0)
# #             pr = d.get('auc_pr', 0)
            
# #             k_strings = []
# #             for k in K_VALUES:
# #                 val_k = get_k_metric(d, metric_type, k)
# #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# #                 k_strings.append(val_str)
            
# #             k_row_str = " | ".join(k_strings)
            
# #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# #         logger.log("\n")

# # def perform_paired_analysis(experiments, logger, filter_loss=None):
# #     paired_data = collections.defaultdict(dict)
# #     for exp in experiments:
# #         if filter_loss and exp['loss'] != filter_loss: continue
# #         uid = (exp['model'], exp['config_id'], exp['run'])
# #         paired_data[uid][exp['selection_metric']] = exp
        
# #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# #     pair_count = 0
    
# #     for uid, metrics in paired_data.items():
# #         if 'prauc' in metrics and 'rocauc' in metrics:
# #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# #             pair_count += 1
# #             for m in METRICS_OF_INTEREST:
# #                 for k in K_VALUES:
# #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# #                     if v_pr is not None and v_roc is not None:
# #                         results[m][k].append((v_pr, v_roc))

# #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
# #     # Updated Header: Replaced "Mean Ratio" with "Agg Ratio" and "Median Ratio"
# #     logger.log("| Metric | Top-K | Agg Ratio (Total/Total) | Median Ratio (Run-wise) | Win Rate (PR > ROC) | P-Value |")
# #     logger.log("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
# #     for m in METRICS_OF_INTEREST:
# #         first = True
# #         for k in K_VALUES:
# #             pairs = results[m][k]
# #             if not pairs: continue
# #             arr_pr = np.array([p[0] for p in pairs])
# #             arr_roc = np.array([p[1] for p in pairs])
            
# #             # 1. Aggregate Ratio (Robust to individual 0s)
# #             # Sum of PR scores / Sum of ROC scores
# #             agg_ratio = np.sum(arr_pr) / (np.sum(arr_roc) + 1e-9)
            
# #             # 2. Median Ratio (Robust to outliers)
# #             # We calculate individual ratios first
# #             individual_ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
# #             median_ratio = np.median(individual_ratios)
            
# #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# #             try: 
# #                 diff = arr_pr - arr_roc
# #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# #             except: p_val = 1.0
            
# #             label = f"**{m.upper()}**" if first else ""
# #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            
# #             agg_str = f"**{agg_ratio:.3f}x**" if agg_ratio > 1.0 else f"{agg_ratio:.3f}x"
# #             med_str = f"**{median_ratio:.3f}x**" if median_ratio > 1.0 else f"{median_ratio:.3f}x"
            
# #             logger.log(f"| {label} | {k}% | {agg_str} | {med_str} | {win:.1f}% | {p_str} |")
# #             first = False
# #         logger.log("| | | | | | |")

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# #     parser.add_argument("--model", type=str, default=None)
# #     parser.add_argument("--filter", type=str, default=None)
# #     args = parser.parse_args()
    
# #     output_filename = f"analysis_report_{args.dataset}"
# #     if args.model: output_filename += f"_{args.model}"
# #     if args.filter: output_filename += f"_{args.filter}"
# #     output_filename += ".md"
    
# #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# #     logger = ReportLogger(out_path)
    
# #     header = f"# Analysis Report for {args.dataset}"
# #     if args.model: header += f" ({args.model})"
# #     if args.filter: header += f" [Filter: {args.filter}]"
# #     logger.log(header)
    
# #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# #     if not exps: 
# #         print("No results found.")
# #         return
    
# #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# #     analyze_part_a(exps, logger)
# #     logger.log(f"\n## Part B: The 'Ratio Test'")
# #     perform_paired_analysis(exps, logger, None)
# #     perform_paired_analysis(exps, logger, 'Standard')
# #     perform_paired_analysis(exps, logger, 'Focal')
    
# #     logger.save()

# # if __name__ == "__main__":
# #     main()




# # # import os
# # # import json
# # # import argparse
# # # import numpy as np
# # # from scipy.stats import wilcoxon
# # # import collections
# # # import warnings
# # # import sys

# # # warnings.filterwarnings("ignore")

# # # # ==============================================================================
# # # # CONFIGURATION
# # # # ==============================================================================

# # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # ==============================================================================
# # # # OUTPUT HELPER
# # # # ==============================================================================

# # # class ReportLogger:
# # #     def __init__(self, filepath):
# # #         self.filepath = filepath
# # #         self.content = []
        
# # #     def log(self, message=""):
# # #         print(message)
# # #         self.content.append(message)
        
# # #     def save(self):
# # #         with open(self.filepath, 'w') as f:
# # #             f.write("\n".join(self.content))
# # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # ==============================================================================
# # # # DATA LOADING
# # # # ==============================================================================

# # # def parse_folder_name(folder_name):
# # #     """Parses folder string to extract parameters."""
# # #     parts = folder_name.split('_')
# # #     params = {}
# # #     params['loss'] = 'Standard'
# # #     params['metric'] = 'rocauc'
    
# # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # #     for part in parts:
# # #         if part.startswith('metric'):
# # #             raw_val = part.replace('metric', '')
# # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # #             params['metric'] = raw_val
    
# # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # #     config_parts.sort()
# # #     params['config_id'] = "_".join(config_parts)
# # #     return params

# # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # #     experiments = []
# # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # #     if not os.path.exists(dataset_dir):
# # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # #         return []

# # #     for model_name in os.listdir(dataset_dir):
# # #         # Model Filter
# # #         if target_model and model_name.lower() != target_model.lower():
# # #             continue

# # #         model_path = os.path.join(dataset_dir, model_name)
# # #         if not os.path.isdir(model_path): continue
        
# # #         for folder_name in os.listdir(model_path):
# # #             # 1. Skip system folders
# # #             folder_path = os.path.join(model_path, folder_name)
# # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # #                 continue
            
# # #             # 2. String Filter (e.g. 'lr-0.0001')
# # #             if filter_str and filter_str not in folder_name:
# # #                 continue
            
# # #             params = parse_folder_name(folder_name)
# # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # #             for item in os.listdir(folder_path):
# # #                 if not item.startswith('run_'): continue
# # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # #                 run_id = item 
# # #                 metrics_data = None
                
# # #                 if has_topk_folder:
# # #                     json_name = f"{run_id}_metrics.json"
# # #                     json_path = os.path.join(topk_results_path, json_name)
# # #                     if os.path.exists(json_path):
# # #                         try:
# # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # #                         except: pass
                
# # #                 if metrics_data is None:
# # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # #                     if os.path.exists(json_path):
# # #                         try:
# # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # #                         except: pass
                
# # #                 if metrics_data:
# # #                     entry = {
# # #                         'model': model_name,
# # #                         'config_id': params['config_id'],
# # #                         'loss': params['loss'],
# # #                         'selection_metric': params['metric'],
# # #                         'run': run_id,
# # #                         'data': metrics_data
# # #                     }
# # #                     experiments.append(entry)
                    
# # #     return experiments

# # # def get_k_metric(data, base_metric, k):
# # #     keys_to_try = [
# # #         f"{base_metric}_at_{k}_pct",
# # #         f"{base_metric}_at_{k:.1f}_pct",
# # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # #     ]
# # #     for key in keys_to_try:
# # #         if key in data: return data[key]
# # #     return None

# # # # ==============================================================================
# # # # ANALYSIS FUNCTIONS
# # # # ==============================================================================

# # # def analyze_part_a(experiments, logger):
# # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# # #     # 1. Group Experiments
# # #     groups = collections.defaultdict(list)
# # #     for exp in experiments:
# # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # #         groups[key].append(exp)
    
# # #     sorted_keys = sorted(groups.keys())

# # #     # 2. Iterate through each Metric type (NDCG, Precision, Recall)
# # #     for metric_type in METRICS_OF_INTEREST:
# # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # #         # Build Header
# # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # #         separator = "".join(sep_cols)
        
# # #         logger.log(header)
# # #         logger.log(separator)
        
# # #         for key in sorted_keys:
# # #             model, loss, sel_metric = key
# # #             exps = groups[key]
            
# # #             # Find Champion based on Validation Score
# # #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# # #             d = best_exp['data']
            
# # #             # Metadata
# # #             val = d.get('best_validation_metric', -1)
# # #             roc = d.get('auc_roc', 0)
# # #             pr = d.get('auc_pr', 0)
            
# # #             # Collect K values
# # #             k_strings = []
# # #             for k in K_VALUES:
# # #                 val_k = get_k_metric(d, metric_type, k)
# # #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# # #                 k_strings.append(val_str)
            
# # #             k_row_str = " | ".join(k_strings)
            
# # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# # #         logger.log("\n")

# # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # #     paired_data = collections.defaultdict(dict)
# # #     for exp in experiments:
# # #         if filter_loss and exp['loss'] != filter_loss: continue
# # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # #         paired_data[uid][exp['selection_metric']] = exp
        
# # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # #     pair_count = 0
    
# # #     for uid, metrics in paired_data.items():
# # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # #             pair_count += 1
# # #             for m in METRICS_OF_INTEREST:
# # #                 for k in K_VALUES:
# # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # #                     if v_pr is not None and v_roc is not None:
# # #                         results[m][k].append((v_pr, v_roc))

# # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
# # #     logger.log("| Metric | Top-K | Mean Ratio (PR/ROC) | Win Rate (PR > ROC) | P-Value |")
# # #     logger.log("| :--- | :--- | :--- | :--- | :--- |")
    
# # #     for m in METRICS_OF_INTEREST:
# # #         first = True
# # #         for k in K_VALUES:
# # #             pairs = results[m][k]
# # #             if not pairs: continue
# # #             arr_pr = np.array([p[0] for p in pairs])
# # #             arr_roc = np.array([p[1] for p in pairs])
            
# # #             ratio = np.mean((arr_pr + 1e-9)/(arr_roc + 1e-9))
# # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # #             try: 
# # #                 diff = arr_pr - arr_roc
# # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # #             except: p_val = 1.0
            
# # #             label = f"**{m.upper()}**" if first else ""
# # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
# # #             r_str = f"**{ratio:.3f}x**" if ratio > 1.0 else f"{ratio:.3f}x"
# # #             logger.log(f"| {label} | {k}% | {r_str} | {win:.1f}% | {p_str} |")
# # #             first = False
# # #         logger.log("| | | | | |")

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # #     parser.add_argument("--model", type=str, default=None, help="Filter by model (e.g. gcn)")
# # #     parser.add_argument("--filter", type=str, default=None, help="Filter by folder name string (e.g. lr-0.0001)")
# # #     args = parser.parse_args()
    
# # #     # Filename includes filter details
# # #     output_filename = f"analysis_report_{args.dataset}"
# # #     if args.model: output_filename += f"_{args.model}"
# # #     if args.filter: output_filename += f"_{args.filter}"
# # #     output_filename += ".md"
    
# # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # #     logger = ReportLogger(out_path)
    
# # #     header = f"# Analysis Report for {args.dataset}"
# # #     if args.model: header += f" ({args.model})"
# # #     if args.filter: header += f" [Filter: {args.filter}]"
# # #     logger.log(header)
    
# # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # #     if not exps: 
# # #         print("No results found matching your criteria.")
# # #         return
    
# # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # #     analyze_part_a(exps, logger)
# # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # #     perform_paired_analysis(exps, logger, None)
# # #     perform_paired_analysis(exps, logger, 'Standard')
# # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # #     logger.save()

# # # if __name__ == "__main__":
# # #     main()




# # # import os
# # # import json
# # # import argparse
# # # import numpy as np
# # # from scipy.stats import wilcoxon
# # # import collections
# # # import warnings
# # # import sys

# # # warnings.filterwarnings("ignore")

# # # # ==============================================================================
# # # # CONFIGURATION
# # # # ==============================================================================

# # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # ==============================================================================
# # # # OUTPUT HELPER
# # # # ==============================================================================

# # # class ReportLogger:
# # #     def __init__(self, filepath):
# # #         self.filepath = filepath
# # #         self.content = []
        
# # #     def log(self, message=""):
# # #         print(message)
# # #         self.content.append(message)
        
# # #     def save(self):
# # #         with open(self.filepath, 'w') as f:
# # #             f.write("\n".join(self.content))
# # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # ==============================================================================
# # # # DATA LOADING
# # # # ==============================================================================

# # # def parse_folder_name(folder_name):
# # #     """Parses folder string to extract parameters."""
# # #     parts = folder_name.split('_')
# # #     params = {}
# # #     params['loss'] = 'Standard'
# # #     params['metric'] = 'rocauc'
    
# # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # #     for part in parts:
# # #         if part.startswith('metric'):
# # #             raw_val = part.replace('metric', '')
# # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # #             params['metric'] = raw_val
    
# # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # #     config_parts.sort()
# # #     params['config_id'] = "_".join(config_parts)
# # #     return params

# # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # #     experiments = []
# # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # #     if not os.path.exists(dataset_dir):
# # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # #         return []

# # #     for model_name in os.listdir(dataset_dir):
# # #         # Model Filter
# # #         if target_model and model_name.lower() != target_model.lower():
# # #             continue

# # #         model_path = os.path.join(dataset_dir, model_name)
# # #         if not os.path.isdir(model_path): continue
        
# # #         for folder_name in os.listdir(model_path):
# # #             # 1. Skip system folders
# # #             folder_path = os.path.join(model_path, folder_name)
# # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # #                 continue
            
# # #             # 2. String Filter (e.g. 'lr-0.0001')
# # #             if filter_str and filter_str not in folder_name:
# # #                 continue
            
# # #             params = parse_folder_name(folder_name)
# # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # #             for item in os.listdir(folder_path):
# # #                 if not item.startswith('run_'): continue
# # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # #                 run_id = item 
# # #                 metrics_data = None
                
# # #                 if has_topk_folder:
# # #                     json_name = f"{run_id}_metrics.json"
# # #                     json_path = os.path.join(topk_results_path, json_name)
# # #                     if os.path.exists(json_path):
# # #                         try:
# # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # #                         except: pass
                
# # #                 if metrics_data is None:
# # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # #                     if os.path.exists(json_path):
# # #                         try:
# # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # #                         except: pass
                
# # #                 if metrics_data:
# # #                     entry = {
# # #                         'model': model_name,
# # #                         'config_id': params['config_id'],
# # #                         'loss': params['loss'],
# # #                         'selection_metric': params['metric'],
# # #                         'run': run_id,
# # #                         'data': metrics_data
# # #                     }
# # #                     experiments.append(entry)
                    
# # #     return experiments

# # # def get_k_metric(data, base_metric, k):
# # #     keys_to_try = [
# # #         f"{base_metric}_at_{k}_pct",
# # #         f"{base_metric}_at_{k:.1f}_pct",
# # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # #     ]
# # #     for key in keys_to_try:
# # #         if key in data: return data[key]
# # #     return None

# # # # ==============================================================================
# # # # ANALYSIS FUNCTIONS
# # # # ==============================================================================

# # # def analyze_part_a(experiments, logger):
# # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# # #     # 1. Group Experiments
# # #     groups = collections.defaultdict(list)
# # #     for exp in experiments:
# # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # #         groups[key].append(exp)
    
# # #     sorted_keys = sorted(groups.keys())

# # #     # 2. Iterate through each Metric type (NDCG, Precision, Recall)
# # #     for metric_type in METRICS_OF_INTEREST:
# # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # #         # Build Header
# # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # #         separator = "".join(sep_cols)
        
# # #         logger.log(header)
# # #         logger.log(separator)
        
# # #         for key in sorted_keys:
# # #             model, loss, sel_metric = key
# # #             exps = groups[key]
            
# # #             # Find Champion based on Validation Score
# # #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# # #             d = best_exp['data']
            
# # #             # Metadata
# # #             val = d.get('best_validation_metric', -1)
# # #             roc = d.get('auc_roc', 0)
# # #             pr = d.get('auc_pr', 0)
            
# # #             # Collect K values
# # #             k_strings = []
# # #             for k in K_VALUES:
# # #                 val_k = get_k_metric(d, metric_type, k)
# # #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# # #                 k_strings.append(val_str)
            
# # #             k_row_str = " | ".join(k_strings)
            
# # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# # #         logger.log("\n")

# # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # #     paired_data = collections.defaultdict(dict)
# # #     for exp in experiments:
# # #         if filter_loss and exp['loss'] != filter_loss: continue
# # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # #         paired_data[uid][exp['selection_metric']] = exp
        
# # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # #     pair_count = 0
    
# # #     for uid, metrics in paired_data.items():
# # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # #             pair_count += 1
# # #             for m in METRICS_OF_INTEREST:
# # #                 for k in K_VALUES:
# # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # #                     if v_pr is not None and v_roc is not None:
# # #                         results[m][k].append((v_pr, v_roc))

# # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
# # #     logger.log("| Metric | Top-K | Mean Ratio (PR/ROC) | Win Rate (PR > ROC) | P-Value |")
# # #     logger.log("| :--- | :--- | :--- | :--- | :--- |")
    
# # #     for m in METRICS_OF_INTEREST:
# # #         first = True
# # #         for k in K_VALUES:
# # #             pairs = results[m][k]
# # #             if not pairs: continue
# # #             arr_pr = np.array([p[0] for p in pairs])
# # #             arr_roc = np.array([p[1] for p in pairs])
            
# # #             ratio = np.mean((arr_pr + 1e-9)/(arr_roc + 1e-9))
# # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # #             try: 
# # #                 diff = arr_pr - arr_roc
# # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # #             except: p_val = 1.0
            
# # #             label = f"**{m.upper()}**" if first else ""
# # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
# # #             r_str = f"**{ratio:.3f}x**" if ratio > 1.0 else f"{ratio:.3f}x"
# # #             logger.log(f"| {label} | {k}% | {r_str} | {win:.1f}% | {p_str} |")
# # #             first = False
# # #         logger.log("| | | | | |")

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # #     parser.add_argument("--model", type=str, default=None, help="Filter by model (e.g. gcn)")
# # #     parser.add_argument("--filter", type=str, default=None, help="Filter by folder name string (e.g. lr-0.0001)")
# # #     args = parser.parse_args()
    
# # #     # Filename includes filter details
# # #     output_filename = f"analysis_report_{args.dataset}"
# # #     if args.model: output_filename += f"_{args.model}"
# # #     if args.filter: output_filename += f"_{args.filter}"
# # #     output_filename += ".md"
    
# # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # #     logger = ReportLogger(out_path)
    
# # #     header = f"# Analysis Report for {args.dataset}"
# # #     if args.model: header += f" ({args.model})"
# # #     if args.filter: header += f" [Filter: {args.filter}]"
# # #     logger.log(header)
    
# # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # #     if not exps: 
# # #         print("No results found matching your criteria.")
# # #         return
    
# # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # #     analyze_part_a(exps, logger)
# # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # #     perform_paired_analysis(exps, logger, None)
# # #     perform_paired_analysis(exps, logger, 'Standard')
# # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # #     logger.save()

# # # if __name__ == "__main__":
# # #     main()