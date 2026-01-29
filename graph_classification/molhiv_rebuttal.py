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
K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
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
# DATA LOADING - Updated for GPS & Subgraphormer Compatibility
# ==============================================================================

def parse_folder_name(folder_name, model_name="gnn"):
    """Parses folder string to extract parameters, handling GNN, GPS, and Subgraphormer naming."""
    parts = folder_name.split('_')
    params = {'loss': 'Standard', 'metric': 'rocauc'}
    
    # Identify Loss
    if 'class_weight' in parts or 'class-weight' in parts: 
        params['loss'] = 'Standard' 
    elif any(p.startswith('focal') for p in parts): 
        params['loss'] = 'Focal'
        
    # Identify Selection Metric
    for part in parts:
        if part.startswith('metric'):
            params['metric'] = part.replace('metric', '').replace('-', '')
            
    # Config ID generation (ID without the metric part for pairing)
    # Subgraphormer and GPS have complex folder names; we strip varying parts to find the "base" config
    if model_name.lower() in ["gps", "subgraphormer"]:
        # Strip selection metric and pooling to pair different selection strategies for the same architecture
        config_parts = sorted([p for p in parts if not p.startswith('metric') and not p.startswith('pool')])
    else:
        config_parts = sorted([p for p in parts if not p.startswith('metric')])
        
    params['config_id'] = "_".join(config_parts)
    return params

def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
    experiments = []
    dataset_dir = os.path.join(base_dir, dataset)
    if not os.path.exists(dataset_dir): return []

    for model_name in os.listdir(dataset_dir):
        # Allow filtering by model (GCN, GAT, SAGE, GIN, GPS, Subgraphormer)
        if target_model and model_name.lower() != target_model.lower(): continue
        
        model_path = os.path.join(dataset_dir, model_name)
        if not os.path.isdir(model_path): continue
        
        for folder_name in os.listdir(model_path):
            folder_path = os.path.join(model_path, folder_name)
            if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results'] or folder_name.startswith('.'):
                continue
            if filter_str and filter_str not in folder_name: continue
            
            params = parse_folder_name(folder_name, model_name=model_name)
            topk_path = os.path.join(folder_path, 'topk_results')
            run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
            for run_id in run_dirs:
                # OGB/GNN standard usually use metrics.json
                orig_json = os.path.join(folder_path, run_id, 'metrics.json')
                new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
                combined_data = {}
                
                # Check if file exists AND is not empty
                if os.path.exists(orig_json) and os.path.getsize(orig_json) > 0:
                    try:
                        with open(orig_json, 'r') as f:
                            combined_data.update(json.load(f))
                    except json.JSONDecodeError: pass
                
                if os.path.exists(new_json) and os.path.getsize(new_json) > 0:
                    try:
                        with open(new_json, 'r') as f:
                            combined_data.update(json.load(f))
                    except json.JSONDecodeError: pass
                
                if combined_data:
                    experiments.append({
                        'model': model_name, 'config_id': params['config_id'],
                        'loss': params['loss'], 'selection_metric': params['metric'],
                        'run': run_id, 'data': combined_data
                    })
    return experiments

def get_k_metric(data, base_metric, k):
    key = f"{base_metric}_at_{k}_pct"
    return data.get(key)

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_part_a(experiments, logger):
    logger.log(f"\n## Part A: Full Spectrum Performance Decay (Mean ± Std)\n")
    groups = collections.defaultdict(list)
    for exp in experiments:
        groups[(exp['model'], exp['loss'], exp['selection_metric'])].append(exp)
    
    subset_k = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))

    for m_type in METRICS_OF_INTEREST:
        logger.log(f"### Metric: {m_type.upper()}")
        header = "| Model | Loss | Selection | Val | ROC | PR | " + " | ".join([f"@{k}%" for k in subset_k]) + " |"
        sep = "| :--- " * (6 + len(subset_k)) + "|"
        logger.log(header); logger.log(sep)
        
        for key in sorted(groups.keys()):
            model, loss, sel = key
            c_groups = collections.defaultdict(list)
            for e in groups[key]: c_groups[e['config_id']].append(e)
            
            # Selection based on mean validation performance
            best_cid = max(c_groups.keys(), key=lambda c: np.mean([e['data'].get('best_validation_metric', e['data'].get('auc_roc', 0)) for e in c_groups[c]]))
            best_runs = c_groups[best_cid]
            
            def get_stat(key_name):
                vals = [e['data'].get(key_name, 0) for e in best_runs]
                return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

            v_str, r_str, p_str = get_stat('best_validation_metric'), get_stat('auc_roc'), get_stat('auc_pr')
            
            k_strs = []
            for k in subset_k:
                vals = [get_k_metric(e['data'], m_type, k) for e in best_runs]
                vals = [v for v in vals if v is not None]
                k_strs.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "-")
            
            logger.log(f"| {model} | {loss} | {sel.upper()} | {v_str} | {r_str} | {p_str} | {' | '.join(k_strs)} |")
        logger.log("\n")

def perform_paired_analysis(experiments, logger):
    paired = collections.defaultdict(dict)
    for exp in experiments:
        uid = (exp['model'], exp['config_id'], exp['run'])
        paired[uid][exp['selection_metric']] = exp
        
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    for uid, metrics in paired.items():
        if 'prauc' in metrics and 'rocauc' in metrics:
            for m in METRICS_OF_INTEREST:
                for k in K_VALUES:
                    v_pr = get_k_metric(metrics['prauc']['data'], m, k)
                    v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
                    if v_pr is not None and v_roc is not None:
                        results[m][k].append((v_pr, v_roc))

    logger.log(f"## Part B: Selection Strategy Impact (PR vs ROC Selection)")
    display_k = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
    
    for m in METRICS_OF_INTEREST:
        logger.log(f"### {m.upper()} Comparison")
        logger.log("| Top-K | Agg Ratio | % Gain | Win Rate | P-Value |")
        logger.log("| :--- | :--- | :--- | :--- | :--- |")
        for k in display_k:
            pairs = results[m][k]
            if not pairs: continue
            a_pr, a_roc = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            agg = np.sum(a_pr) / (np.sum(a_roc) + 1e-9)
            gain = (np.median((a_pr + 1e-9) / (a_roc + 1e-9)) - 1) * 100
            win = np.sum(a_pr > a_roc) / len(pairs) * 100
            try: p_val = wilcoxon(a_pr, a_roc)[1] if not np.all(a_pr == a_roc) else 1.0
            except: p_val = 1.0
            logger.log(f"| {k}% | {agg:.3f}x | {gain:+.1f}% | {win:.1f}% | {p_val:.2e} |")
        logger.log("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results_weighted")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None) 
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()
    
    exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
    if not exps: 
        print("No experiments found matching criteria.")
        return
    
    fname_parts = ["report", args.dataset]
    if args.model: fname_parts.append(args.model)
    if args.filter: fname_parts.append(args.filter.replace("-", "_"))
    
    output_name = "_".join(fname_parts) + ".md"
    out_path = os.path.join(args.base_dir, args.dataset, output_name)
    
    logger = ReportLogger(out_path)
    logger.log(f"# Rebuttal Analysis: {args.dataset}")
    analyze_part_a(exps, logger)
    perform_paired_analysis(exps, logger)
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

# warnings.filterwarnings("ignore")

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================
# K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
# METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

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
#         print(f"\n[Report saved to: {self.filepath}]")

# # ==============================================================================
# # DATA LOADING - Updated for GPS Compatibility
# # ==============================================================================

# def parse_folder_name(folder_name, is_gps=False):
#     """Parses folder string to extract parameters, handling GNN and GPS naming."""
#     parts = folder_name.split('_')
#     params = {'loss': 'Standard', 'metric': 'rocauc'}
    
#     # Identify Loss
#     if 'class_weight' in parts or 'class-weight' in parts: 
#         params['loss'] = 'Standard' 
#     elif any(p.startswith('focal') for p in parts): 
#         params['loss'] = 'Focal'
        
#     # Identify Selection Metric
#     for part in parts:
#         if part.startswith('metric'):
#             params['metric'] = part.replace('metric', '').replace('-', '')
            
#     # Config ID generation (ID without the metric part for pairing)
#     if is_gps:
#         # GPS folders are long, we strip metric and pool to create a base config ID
#         config_parts = sorted([p for p in parts if not p.startswith('metric') and not p.startswith('pool')])
#     else:
#         config_parts = sorted([p for p in parts if not p.startswith('metric')])
        
#     params['config_id'] = "_".join(config_parts)
#     return params

# def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
#     experiments = []
#     dataset_dir = os.path.join(base_dir, dataset)
#     if not os.path.exists(dataset_dir): return []

#     for model_name in os.listdir(dataset_dir):
#         # Allow filtering by model (GCN, GAT, SAGE, GIN, GPS)
#         if target_model and model_name.lower() != target_model.lower(): continue
        
#         model_path = os.path.join(dataset_dir, model_name)
#         if not os.path.isdir(model_path): continue
        
#         is_gps = (model_name.upper() == "GPS")
        
#         for folder_name in os.listdir(model_path):
#             folder_path = os.path.join(model_path, folder_name)
#             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results'] or folder_name.startswith('.'):
#                 continue
#             if filter_str and filter_str not in folder_name: continue
            
#             params = parse_folder_name(folder_name, is_gps=is_gps)
#             topk_path = os.path.join(folder_path, 'topk_results')
#             run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
#             for run_id in run_dirs:
#                 # Training script used 'best_model.pt', but metrics are in metrics.json
#                 orig_json = os.path.join(folder_path, run_id, 'metrics.json')
#                 new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
#                 combined_data = {}
                
#                 # Check if file exists AND is not empty (Fix for GAT error)
#                 if os.path.exists(orig_json) and os.path.getsize(orig_json) > 0:
#                     try:
#                         with open(orig_json, 'r') as f:
#                             combined_data.update(json.load(f))
#                     except json.JSONDecodeError: pass
                
#                 if os.path.exists(new_json) and os.path.getsize(new_json) > 0:
#                     try:
#                         with open(new_json, 'r') as f:
#                             combined_data.update(json.load(f))
#                     except json.JSONDecodeError: pass
                
#                 if combined_data:
#                     experiments.append({
#                         'model': model_name, 'config_id': params['config_id'],
#                         'loss': params['loss'], 'selection_metric': params['metric'],
#                         'run': run_id, 'data': combined_data
#                     })
#     return experiments

# def get_k_metric(data, base_metric, k):
#     key = f"{base_metric}_at_{k}_pct"
#     return data.get(key)

# # ==============================================================================
# # ANALYSIS FUNCTIONS
# # ==============================================================================

# def analyze_part_a(experiments, logger):
#     logger.log(f"\n## Part A: Full Spectrum Performance Decay (Mean ± Std)\n")
#     groups = collections.defaultdict(list)
#     for exp in experiments:
#         groups[(exp['model'], exp['loss'], exp['selection_metric'])].append(exp)
    
#     # We display a subset for the MD table to keep it readable, but you have all data
#     subset_k = [0.5, 1, 5, 10, 25, 50, 100]

#     for m_type in METRICS_OF_INTEREST:
#         logger.log(f"### Metric: {m_type.upper()}")
#         header = "| Model | Loss | Selection | Val | ROC | PR | " + " | ".join([f"@{k}%" for k in subset_k]) + " |"
#         sep = "| :--- " * (6 + len(subset_k)) + "|"
#         logger.log(header); logger.log(sep)
        
#         for key in sorted(groups.keys()):
#             model, loss, sel = key
#             c_groups = collections.defaultdict(list)
#             for e in groups[key]: c_groups[e['config_id']].append(e)
            
#             # Select best config by mean validation/roc
#             best_cid = max(c_groups.keys(), key=lambda c: np.mean([e['data'].get('best_validation_metric', e['data'].get('auc_roc', 0)) for e in c_groups[c]]))
#             best_runs = c_groups[best_cid]
            
#             def get_stat(key_name):
#                 vals = [e['data'].get(key_name, 0) for e in best_runs]
#                 return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

#             v_str, r_str, p_str = get_stat('best_validation_metric'), get_stat('auc_roc'), get_stat('auc_pr')
            
#             k_strs = []
#             for k in subset_k:
#                 vals = [get_k_metric(e['data'], m_type, k) for e in best_runs]
#                 vals = [v for v in vals if v is not None]
#                 k_strs.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "-")
            
#             logger.log(f"| {model} | {loss} | {sel.upper()} | {v_str} | {r_str} | {p_str} | {' | '.join(k_strs)} |")
#         logger.log("\n")

# def perform_paired_analysis(experiments, logger):
#     paired = collections.defaultdict(dict)
#     for exp in experiments:
#         # Pairing logic: (Model, Base Architecture/Params, Run Number)
#         uid = (exp['model'], exp['config_id'], exp['run'])
#         paired[uid][exp['selection_metric']] = exp
        
#     results = collections.defaultdict(lambda: collections.defaultdict(list))
#     for uid, metrics in paired.items():
#         if 'prauc' in metrics and 'rocauc' in metrics:
#             for m in METRICS_OF_INTEREST:
#                 for k in K_VALUES:
#                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
#                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
#                     if v_pr is not None and v_roc is not None:
#                         results[m][k].append((v_pr, v_roc))

#     logger.log(f"## Part B: Selection Strategy Impact (PR vs ROC Selection)")
#     # Show checkpoints for the rebuttal table
#     display_k = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
    
#     for m in METRICS_OF_INTEREST:
#         logger.log(f"### {m.upper()} Comparison")
#         logger.log("| Top-K | Agg Ratio | % Gain | Win Rate | P-Value |")
#         logger.log("| :--- | :--- | :--- | :--- | :--- |")
#         for k in display_k:
#             pairs = results[m][k]
#             if not pairs: continue
#             a_pr, a_roc = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
#             agg = np.sum(a_pr) / (np.sum(a_roc) + 1e-9)
#             gain = (np.median((a_pr + 1e-9) / (a_roc + 1e-9)) - 1) * 100
#             win = np.sum(a_pr > a_roc) / len(pairs) * 100
#             try: p_val = wilcoxon(a_pr, a_roc)[1] if not np.all(a_pr == a_roc) else 1.0
#             except: p_val = 1.0
#             logger.log(f"| {k}% | {agg:.3f}x | {gain:+.1f}% | {win:.1f}% | {p_val:.2e} |")
#         logger.log("\n")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_dir", type=str, default="results_weighted")
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--model", type=str, default=None) # e.g., GPS or GCN
#     parser.add_argument("--filter", type=str, default=None)
#     args = parser.parse_args()
    
#     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
#     if not exps: return
    
#     fname_parts = ["report", args.dataset]
#     if args.model: fname_parts.append(args.model)
#     if args.filter: fname_parts.append(args.filter.replace("-", "_"))
    
#     output_name = "_".join(fname_parts) + ".md"
#     out_path = os.path.join(args.base_dir, args.dataset, output_name)
    
#     logger = ReportLogger(out_path)
#     logger.log(f"# Rebuttal Analysis: {args.dataset}")
#     analyze_part_a(exps, logger)
#     perform_paired_analysis(exps, logger)
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

# # warnings.filterwarnings("ignore")

# # # ==============================================================================
# # # CONFIGURATION
# # # ==============================================================================
# # K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
# # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

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
# #         print(f"\n[Report saved to: {self.filepath}]")

# # # ==============================================================================
# # # DATA LOADING
# # # ==============================================================================

# # def parse_folder_name(folder_name):
# #     parts = folder_name.split('_')
# #     params = {'loss': 'Standard', 'metric': 'rocauc'}
# #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
# #     for part in parts:
# #         if part.startswith('metric'):
# #             params['metric'] = part.replace('metric', '').replace('-', '')
# #     config_parts = sorted([p for p in parts if not p.startswith('metric')])
# #     params['config_id'] = "_".join(config_parts)
# #     return params

# # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# #     experiments = []
# #     dataset_dir = os.path.join(base_dir, dataset)
# #     if not os.path.exists(dataset_dir): return []

# #     for model_name in os.listdir(dataset_dir):
# #         if target_model and model_name.lower() != target_model.lower(): continue
# #         model_path = os.path.join(dataset_dir, model_name)
# #         if not os.path.isdir(model_path): continue
        
# #         for folder_name in os.listdir(model_path):
# #             folder_path = os.path.join(model_path, folder_name)
# #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results']: continue
# #             if filter_str and filter_str not in folder_name: continue
            
# #             params = parse_folder_name(folder_name)
# #             topk_path = os.path.join(folder_path, 'topk_results')
# #             run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
# #             for run_id in run_dirs:
# #                 orig_json = os.path.join(folder_path, run_id, 'metrics.json')
# #                 new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
# #                 combined_data = {}
                
# #                 # --- FIX: Check if file exists AND is not empty ---
# #                 if os.path.exists(orig_json) and os.path.getsize(orig_json) > 0:
# #                     try:
# #                         with open(orig_json, 'r') as f:
# #                             combined_data.update(json.load(f))
# #                     except json.JSONDecodeError:
# #                         print(f"Warning: Corrupt JSON in {orig_json}. Skipping.")
                
# #                 if os.path.exists(new_json) and os.path.getsize(new_json) > 0:
# #                     try:
# #                         with open(new_json, 'r') as f:
# #                             combined_data.update(json.load(f))
# #                     except json.JSONDecodeError:
# #                         print(f"Warning: Corrupt JSON in {new_json}. Skipping.")
                
# #                 if combined_data:
# #                     experiments.append({
# #                         'model': model_name, 'config_id': params['config_id'],
# #                         'loss': params['loss'], 'selection_metric': params['metric'],
# #                         'run': run_id, 'data': combined_data
# #                     })
# #     return experiments

# # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # #     experiments = []
# # #     dataset_dir = os.path.join(base_dir, dataset)
# # #     if not os.path.exists(dataset_dir): return []

# # #     for model_name in os.listdir(dataset_dir):
# # #         if target_model and model_name.lower() != target_model.lower(): continue
# # #         model_path = os.path.join(dataset_dir, model_name)
# # #         if not os.path.isdir(model_path): continue
        
# # #         for folder_name in os.listdir(model_path):
# # #             folder_path = os.path.join(model_path, folder_name)
# # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results']: continue
# # #             if filter_str and filter_str not in folder_name: continue
            
# # #             params = parse_folder_name(folder_name)
# # #             topk_path = os.path.join(folder_path, 'topk_results')
# # #             run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
# # #             for run_id in run_dirs:
# # #                 orig_json = os.path.join(folder_path, run_id, 'metrics.json')
# # #                 new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
# # #                 combined_data = {}
# # #                 if os.path.exists(orig_json):
# # #                     with open(orig_json, 'r') as f: combined_data.update(json.load(f))
# # #                 if os.path.exists(new_json):
# # #                     with open(new_json, 'r') as f: combined_data.update(json.load(f))
                
# # #                 if combined_data:
# # #                     experiments.append({
# # #                         'model': model_name, 'config_id': params['config_id'],
# # #                         'loss': params['loss'], 'selection_metric': params['metric'],
# # #                         'run': run_id, 'data': combined_data
# # #                     })
# # #     return experiments

# # def get_k_metric(data, base_metric, k):
# #     key = f"{base_metric}_at_{k}_pct"
# #     return data.get(key)

# # # ==============================================================================
# # # ANALYSIS
# # # ==============================================================================

# # def analyze_part_a(experiments, logger):
# #     logger.log(f"\n## Part A: Full Spectrum Performance Decay (Mean ± Std)\n")
# #     groups = collections.defaultdict(list)
# #     for exp in experiments:
# #         groups[(exp['model'], exp['loss'], exp['selection_metric'])].append(exp)
    
# #     for m_type in METRICS_OF_INTEREST:
# #         logger.log(f"### Metric: {m_type.upper()}")
# #         header = "| Model | Loss | Selection | Val | ROC | PR | " + " | ".join([f"@{k}%" for k in K_VALUES]) + " |"
# #         sep = "| :--- " * (6 + len(K_VALUES)) + "|"
# #         logger.log(header); logger.log(sep)
        
# #         for key in sorted(groups.keys()):
# #             model, loss, sel = key
# #             c_groups = collections.defaultdict(list)
# #             for e in groups[key]: c_groups[e['config_id']].append(e)
            
# #             # Selection based on mean validation performance
# #             best_cid = max(c_groups.keys(), key=lambda c: np.mean([e['data'].get('best_validation_metric', e['data'].get('auc_roc', 0)) for e in c_groups[c]]))
# #             best_runs = c_groups[best_cid]
            
# #             # Helper for mean/std strings
# #             def get_stat(key_name):
# #                 vals = [e['data'].get(key_name, 0) for e in best_runs]
# #                 return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

# #             v_str = get_stat('best_validation_metric')
# #             r_str = get_stat('auc_roc')
# #             p_str = get_stat('auc_pr')
            
# #             k_strs = []
# #             for k in K_VALUES:
# #                 vals = [get_k_metric(e['data'], m_type, k) for e in best_runs]
# #                 vals = [val for val in vals if val is not None]
# #                 if vals:
# #                     k_strs.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
# #                 else:
# #                     k_strs.append("-")
            
# #             logger.log(f"| {model} | {loss} | {sel.upper()} | {v_str} | {r_str} | {p_str} | {' | '.join(k_strs)} |")
# #         logger.log("\n")

# # def perform_paired_analysis(experiments, logger):
# #     paired = collections.defaultdict(dict)
# #     for exp in experiments:
# #         uid = (exp['model'], exp['config_id'], exp['run'])
# #         paired[uid][exp['selection_metric']] = exp
        
# #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# #     for uid, metrics in paired.items():
# #         if 'prauc' in metrics and 'rocauc' in metrics:
# #             for m in METRICS_OF_INTEREST:
# #                 for k in K_VALUES:
# #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# #                     if v_pr is not None and v_roc is not None:
# #                         results[m][k].append((v_pr, v_roc))

# #     logger.log(f"## Part B: Selection Strategy Impact (PR vs ROC)")
    
# #     for m in METRICS_OF_INTEREST:
# #         logger.log(f"### {m.upper()} Comparison (All Percentages)")
# #         logger.log("| Top-K | Agg Ratio | % Gain | Win Rate | P-Value |")
# #         logger.log("| :--- | :--- | :--- | :--- | :--- |")
# #         for k in K_VALUES:
# #             pairs = results[m][k]
# #             if not pairs: continue
# #             a_pr, a_roc = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
# #             agg = np.sum(a_pr) / (np.sum(a_roc) + 1e-9)
# #             gain = (np.median((a_pr + 1e-9) / (a_roc + 1e-9)) - 1) * 100
# #             win = np.sum(a_pr > a_roc) / len(pairs) * 100
# #             try: p_val = wilcoxon(a_pr, a_roc)[1] if not np.all(a_pr == a_roc) else 1.0
# #             except: p_val = 1.0
# #             logger.log(f"| {k}% | {agg:.3f}x | {gain:+.1f}% | {win:.1f}% | {p_val:.2e} |")
# #         logger.log("\n")

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# #     parser.add_argument("--dataset", type=str, required=True)
# #     parser.add_argument("--model", type=str, default=None)
# #     parser.add_argument("--filter", type=str, default=None)
# #     args = parser.parse_args()
    
# #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# #     if not exps: return
    
# #     fname_parts = ["report", args.dataset]
# #     if args.model: fname_parts.append(args.model)
# #     if args.filter: fname_parts.append(args.filter.replace("-", "_"))
    
# #     output_name = "_".join(fname_parts) + ".md"
# #     out_path = os.path.join(args.base_dir, args.dataset, output_name)
    
# #     logger = ReportLogger(out_path)
# #     logger.log(f"# Analysis Report: {args.dataset}")
# #     analyze_part_a(exps, logger)
# #     perform_paired_analysis(exps, logger)
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

# # # warnings.filterwarnings("ignore")

# # # # ==============================================================================
# # # # CONFIGURATION - Matches the 5% gap logic
# # # # ==============================================================================
# # # K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
# # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

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
# # #         print(f"\n[Report saved to: {self.filepath}]")

# # # # ==============================================================================
# # # # DATA LOADING
# # # # ==============================================================================

# # # def parse_folder_name(folder_name):
# # #     parts = folder_name.split('_')
# # #     params = {'loss': 'Standard', 'metric': 'rocauc'}
# # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
# # #     for part in parts:
# # #         if part.startswith('metric'):
# # #             params['metric'] = part.replace('metric', '').replace('-', '')
# # #     config_parts = sorted([p for p in parts if not p.startswith('metric')])
# # #     params['config_id'] = "_".join(config_parts)
# # #     return params

# # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # #     experiments = []
# # #     dataset_dir = os.path.join(base_dir, dataset)
# # #     if not os.path.exists(dataset_dir): return []

# # #     for model_name in os.listdir(dataset_dir):
# # #         if target_model and model_name.lower() != target_model.lower(): continue
# # #         model_path = os.path.join(dataset_dir, model_name)
# # #         if not os.path.isdir(model_path): continue
        
# # #         for folder_name in os.listdir(model_path):
# # #             folder_path = os.path.join(model_path, folder_name)
# # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results']: continue
# # #             if filter_str and filter_str not in folder_name: continue
            
# # #             params = parse_folder_name(folder_name)
# # #             topk_path = os.path.join(folder_path, 'topk_results')
# # #             run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
# # #             for run_id in run_dirs:
# # #                 orig_json = os.path.join(folder_path, run_id, 'metrics.json')
# # #                 new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
# # #                 combined_data = {}
# # #                 if os.path.exists(orig_json):
# # #                     with open(orig_json, 'r') as f: combined_data.update(json.load(f))
# # #                 if os.path.exists(new_json):
# # #                     with open(new_json, 'r') as f: combined_data.update(json.load(f))
                
# # #                 if combined_data:
# # #                     experiments.append({
# # #                         'model': model_name, 'config_id': params['config_id'],
# # #                         'loss': params['loss'], 'selection_metric': params['metric'],
# # #                         'run': run_id, 'data': combined_data
# # #                     })
# # #     return experiments

# # # def get_k_metric(data, base_metric, k):
# # #     key = f"{base_metric}_at_{k}_pct"
# # #     return data.get(key)

# # # # ==============================================================================
# # # # ANALYSIS
# # # # ==============================================================================

# # # def analyze_part_a(experiments, logger):
# # #     logger.log(f"\n## Part A: Full Spectrum Performance Decay\n")
# # #     groups = collections.defaultdict(list)
# # #     for exp in experiments:
# # #         groups[(exp['model'], exp['loss'], exp['selection_metric'])].append(exp)
    
# # #     # Use ALL K_VALUES for the table
# # #     for m_type in METRICS_OF_INTEREST:
# # #         logger.log(f"### Metric: {m_type.upper()}")
# # #         # Creating a very wide table - note that Markdown viewers might require scrolling
# # #         header = "| Model | Loss | Selection | Val | ROC | PR | " + " | ".join([f"@{k}%" for k in K_VALUES]) + " |"
# # #         sep = "| :--- " * (6 + len(K_VALUES)) + "|"
# # #         logger.log(header); logger.log(sep)
        
# # #         for key in sorted(groups.keys()):
# # #             model, loss, sel = key
# # #             c_groups = collections.defaultdict(list)
# # #             for e in groups[key]: c_groups[e['config_id']].append(e)
            
# # #             best_cid = max(c_groups.keys(), key=lambda c: np.mean([e['data'].get('best_validation_metric', e['data'].get('auc_roc', 0)) for e in c_groups[c]]))
# # #             best_runs = c_groups[best_cid]
            
# # #             v = np.mean([e['data'].get('best_validation_metric', 0) for e in best_runs])
# # #             r = np.mean([e['data'].get('auc_roc', 0) for e in best_runs])
# # #             p = np.mean([e['data'].get('auc_pr', 0) for e in best_runs])
            
# # #             k_strs = []
# # #             for k in K_VALUES:
# # #                 vals = [get_k_metric(e['data'], m_type, k) for e in best_runs]
# # #                 vals = [val for val in vals if val is not None]
# # #                 k_strs.append(f"{np.mean(vals):.3f}" if vals else "-")
            
# # #             logger.log(f"| {model} | {loss} | {sel.upper()} | {v:.3f} | {r:.3f} | {p:.3f} | {' | '.join(k_strs)} |")
# # #         logger.log("\n")

# # # def perform_paired_analysis(experiments, logger):
# # #     paired = collections.defaultdict(dict)
# # #     for exp in experiments:
# # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # #         paired[uid][exp['selection_metric']] = exp
        
# # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # #     for uid, metrics in paired.items():
# # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # #             for m in METRICS_OF_INTEREST:
# # #                 for k in K_VALUES:
# # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # #                     if v_pr is not None and v_roc is not None:
# # #                         results[m][k].append((v_pr, v_roc))

# # #     logger.log(f"## Part B: Selection Strategy Impact (PR vs ROC)")
    
# # #     for m in METRICS_OF_INTEREST:
# # #         logger.log(f"### {m.upper()} Comparison (All Percentages)")
# # #         logger.log("| Top-K | Agg Ratio | % Gain | Win Rate | P-Value |")
# # #         logger.log("| :--- | :--- | :--- | :--- | :--- |")
# # #         # Use ALL K_VALUES here as rows
# # #         for k in K_VALUES:
# # #             pairs = results[m][k]
# # #             if not pairs: continue
# # #             a_pr, a_roc = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
# # #             agg = np.sum(a_pr) / (np.sum(a_roc) + 1e-9)
# # #             gain = (np.median((a_pr + 1e-9) / (a_roc + 1e-9)) - 1) * 100
# # #             win = np.sum(a_pr > a_roc) / len(pairs) * 100
# # #             try: p_val = wilcoxon(a_pr, a_roc)[1] if not np.all(a_pr == a_roc) else 1.0
# # #             except: p_val = 1.0
# # #             logger.log(f"| {k}% | {agg:.3f}x | {gain:+.1f}% | {win:.1f}% | {p_val:.2e} |")
# # #         logger.log("\n")

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # #     parser.add_argument("--dataset", type=str, required=True)
# # #     parser.add_argument("--model", type=str, default=None)
# # #     parser.add_argument("--filter", type=str, default=None)
# # #     args = parser.parse_args()
    
# # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # #     if not exps: 
# # #         print("No experiments found matching criteria.")
# # #         return
    
# # #     # --- UPDATED FILENAME LOGIC ---
# # #     # Construct filename based on arguments to prevent overwriting
# # #     fname_parts = ["report", args.dataset]
# # #     if args.model: fname_parts.append(args.model)
# # #     if args.filter: fname_parts.append(args.filter.replace("-", "_"))
    
# # #     output_name = "_".join(fname_parts) + ".md"
# # #     out_path = os.path.join(args.base_dir, args.dataset, output_name)
    
# # #     logger = ReportLogger(out_path)
# # #     logger.log(f"# Analysis Report: {args.dataset}")
# # #     if args.model: logger.log(f"**Model:** {args.model}")
# # #     if args.filter: logger.log(f"**Filter:** {args.filter}")
    
# # #     analyze_part_a(exps, logger)
# # #     perform_paired_analysis(exps, logger)
# # #     logger.save()

# # # if __name__ == "__main__":
# # #     main()





# # # # import os
# # # # import json
# # # # import argparse
# # # # import numpy as np
# # # # from scipy.stats import wilcoxon
# # # # import collections
# # # # import warnings

# # # # warnings.filterwarnings("ignore")

# # # # # ==============================================================================
# # # # # CONFIGURATION
# # # # # ==============================================================================
# # # # K_VALUES = [0.5, 1, 2, 3, 4, 5] + list(range(10, 105, 5))
# # # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # class ReportLogger:
# # # #     def __init__(self, filepath):
# # # #         self.filepath = filepath
# # # #         self.content = []
# # # #     def log(self, message=""):
# # # #         print(message)
# # # #         self.content.append(message)
# # # #     def save(self):
# # # #         with open(self.filepath, 'w') as f:
# # # #             f.write("\n".join(self.content))

# # # # # ==============================================================================
# # # # # DATA LOADING - FIXED PATH RESOLUTION
# # # # # ==============================================================================

# # # # def parse_folder_name(folder_name):
# # # #     parts = folder_name.split('_')
# # # #     params = {'loss': 'Standard', 'metric': 'rocauc'}
# # # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
# # # #     for part in parts:
# # # #         if part.startswith('metric'):
# # # #             params['metric'] = part.replace('metric', '').replace('-', '')
# # # #     config_parts = sorted([p for p in parts if not p.startswith('metric')])
# # # #     params['config_id'] = "_".join(config_parts)
# # # #     return params

# # # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # # #     experiments = []
# # # #     dataset_dir = os.path.join(base_dir, dataset)
# # # #     if not os.path.exists(dataset_dir): return []

# # # #     for model_name in os.listdir(dataset_dir):
# # # #         if target_model and model_name.lower() != target_model.lower(): continue
# # # #         model_path = os.path.join(dataset_dir, model_name)
# # # #         if not os.path.isdir(model_path): continue
        
# # # #         for folder_name in os.listdir(model_path):
# # # #             folder_path = os.path.join(model_path, folder_name)
# # # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_results']: continue
# # # #             if filter_str and filter_str not in folder_name: continue
            
# # # #             params = parse_folder_name(folder_name)
# # # #             topk_path = os.path.join(folder_path, 'topk_results')
            
# # # #             # Find all run directories
# # # #             run_dirs = [d for d in os.listdir(folder_path) if d.startswith('run_') and os.path.isdir(os.path.join(folder_path, d))]
            
# # # #             for run_id in run_dirs:
# # # #                 # 1. Load original training metrics (Val AUC, ROC-AUC, etc.)
# # # #                 orig_json = os.path.join(folder_path, run_id, 'metrics.json')
# # # #                 # 2. Load the new Top-K metrics
# # # #                 new_json = os.path.join(topk_path, f"{run_id}_metrics.json")
                
# # # #                 combined_data = {}
# # # #                 if os.path.exists(orig_json):
# # # #                     with open(orig_json, 'r') as f: combined_data.update(json.load(f))
# # # #                 if os.path.exists(new_json):
# # # #                     with open(new_json, 'r') as f: combined_data.update(json.load(f))
                
# # # #                 if combined_data:
# # # #                     experiments.append({
# # # #                         'model': model_name, 'config_id': params['config_id'],
# # # #                         'loss': params['loss'], 'selection_metric': params['metric'],
# # # #                         'run': run_id, 'data': combined_data
# # # #                     })
# # # #     return experiments

# # # # def get_k_metric(data, base_metric, k):
# # # #     # Matches the exact formatting of the evaluation script
# # # #     key = f"{base_metric}_at_{k}_pct"
# # # #     return data.get(key)

# # # # # ==============================================================================
# # # # # ANALYSIS
# # # # # ==============================================================================

# # # # def analyze_part_a(experiments, logger):
# # # #     logger.log(f"\n## Part A: Performance Decay (Subset View)\n")
# # # #     groups = collections.defaultdict(list)
# # # #     for exp in experiments:
# # # #         groups[(exp['model'], exp['loss'], exp['selection_metric'])].append(exp)
    
# # # #     subset_k = [1, 5, 10, 25, 50, 100]

# # # #     for m_type in METRICS_OF_INTEREST:
# # # #         logger.log(f"### Metric: {m_type.upper()}")
# # # #         header = "| Model | Loss | Selection | Val | ROC | PR | " + " | ".join([f"@{k}%" for k in subset_k]) + " |"
# # # #         sep = "| :--- " * (6 + len(subset_k)) + "|"
# # # #         logger.log(header); logger.log(sep)
        
# # # #         for key in sorted(groups.keys()):
# # # #             model, loss, sel = key
# # # #             # Group by config to find best
# # # #             c_groups = collections.defaultdict(list)
# # # #             for e in groups[key]: c_groups[e['config_id']].append(e)
            
# # # #             # Use 'best_validation_metric' or fallback to 'auc_roc' for selection
# # # #             best_cid = max(c_groups.keys(), key=lambda c: np.mean([e['data'].get('best_validation_metric', e['data'].get('auc_roc', 0)) for e in c_groups[c]]))
# # # #             best_runs = c_groups[best_cid]
            
# # # #             v = np.mean([e['data'].get('best_validation_metric', 0) for e in best_runs])
# # # #             r = np.mean([e['data'].get('auc_roc', 0) for e in best_runs])
# # # #             p = np.mean([e['data'].get('auc_pr', 0) for e in best_runs])
            
# # # #             k_strs = []
# # # #             for k in subset_k:
# # # #                 vals = [get_k_metric(e['data'], m_type, k) for e in best_runs]
# # # #                 vals = [v for v in vals if v is not None]
# # # #                 k_strs.append(f"{np.mean(vals):.3f}" if vals else "-")
            
# # # #             logger.log(f"| {model} | {loss} | {sel.upper()} | {v:.3f} | {r:.3f} | {p:.3f} | {' | '.join(k_strs)} |")
# # # #         logger.log("\n")

# # # # def perform_paired_analysis(experiments, logger):
# # # #     paired = collections.defaultdict(dict)
# # # #     for exp in experiments:
# # # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # # #         paired[uid][exp['selection_metric']] = exp
        
# # # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # # #     for uid, metrics in paired.items():
# # # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # # #             for m in METRICS_OF_INTEREST:
# # # #                 for k in K_VALUES:
# # # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # # #                     if v_pr is not None and v_roc is not None:
# # # #                         results[m][k].append((v_pr, v_roc))

# # # #     logger.log(f"## Part B: Selection Strategy Impact (PR vs ROC)")
# # # #     display_k = [0.5, 1, 5, 10, 20, 50, 100]
    
# # # #     for m in METRICS_OF_INTEREST:
# # # #         logger.log(f"### {m.upper()} Comparison")
# # # #         logger.log("| Top-K | Agg Ratio | % Gain | Win Rate | P-Value |")
# # # #         logger.log("| :--- | :--- | :--- | :--- | :--- |")
# # # #         for k in display_k:
# # # #             pairs = results[m][k]
# # # #             if not pairs: continue
# # # #             a_pr, a_roc = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
# # # #             agg = np.sum(a_pr) / (np.sum(a_roc) + 1e-9)
# # # #             gain = (np.median((a_pr + 1e-9) / (a_roc + 1e-9)) - 1) * 100
# # # #             win = np.sum(a_pr > a_roc) / len(pairs) * 100
# # # #             try: p_val = wilcoxon(a_pr, a_roc)[1] if not np.all(a_pr == a_roc) else 1.0
# # # #             except: p_val = 1.0
# # # #             logger.log(f"| {k}% | {agg:.3f}x | {gain:+.1f}% | {win:.1f}% | {p_val:.2e} |")
# # # #         logger.log("\n")

# # # # def main():
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # # #     parser.add_argument("--dataset", type=str, required=True)
# # # #     parser.add_argument("--model", type=str, default=None)
# # # #     parser.add_argument("--filter", type=str, default=None)
# # # #     args = parser.parse_args()
    
# # # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # # #     if not exps: return
    
# # # #     out_path = os.path.join(args.base_dir, args.dataset, f"report_{args.dataset}.md")
# # # #     logger = ReportLogger(out_path)
# # # #     analyze_part_a(exps, logger)
# # # #     perform_paired_analysis(exps, logger)
# # # #     logger.save()

# # # # if __name__ == "__main__":
# # # #     main()

# # # # # import os
# # # # # import json
# # # # # import argparse
# # # # # import numpy as np
# # # # # from scipy.stats import wilcoxon
# # # # # import collections
# # # # # import warnings
# # # # # import sys

# # # # # warnings.filterwarnings("ignore")

# # # # # # ==============================================================================
# # # # # # CONFIGURATION
# # # # # # ==============================================================================

# # # # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # # # ==============================================================================
# # # # # # OUTPUT HELPER
# # # # # # ==============================================================================

# # # # # class ReportLogger:
# # # # #     def __init__(self, filepath):
# # # # #         self.filepath = filepath
# # # # #         self.content = []
        
# # # # #     def log(self, message=""):
# # # # #         print(message)
# # # # #         self.content.append(message)
        
# # # # #     def save(self):
# # # # #         with open(self.filepath, 'w') as f:
# # # # #             f.write("\n".join(self.content))
# # # # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # # # ==============================================================================
# # # # # # DATA LOADING
# # # # # # ==============================================================================

# # # # # def parse_folder_name(folder_name):
# # # # #     """Parses folder string to extract parameters."""
# # # # #     parts = folder_name.split('_')
# # # # #     params = {}
# # # # #     params['loss'] = 'Standard'
# # # # #     params['metric'] = 'rocauc'
    
# # # # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # # # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # # # #     for part in parts:
# # # # #         if part.startswith('metric'):
# # # # #             raw_val = part.replace('metric', '')
# # # # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # # # #             params['metric'] = raw_val
    
# # # # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # # # #     config_parts.sort()
# # # # #     params['config_id'] = "_".join(config_parts)
# # # # #     return params

# # # # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # # # #     experiments = []
# # # # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # # # #     if not os.path.exists(dataset_dir):
# # # # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # # # #         return []

# # # # #     for model_name in os.listdir(dataset_dir):
# # # # #         if target_model and model_name.lower() != target_model.lower():
# # # # #             continue

# # # # #         model_path = os.path.join(dataset_dir, model_name)
# # # # #         if not os.path.isdir(model_path): continue
        
# # # # #         for folder_name in os.listdir(model_path):
# # # # #             folder_path = os.path.join(model_path, folder_name)
# # # # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # # # #                 continue
            
# # # # #             if filter_str and filter_str not in folder_name:
# # # # #                 continue
            
# # # # #             params = parse_folder_name(folder_name)
# # # # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # # # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # # # #             for item in os.listdir(folder_path):
# # # # #                 if not item.startswith('run_'): continue
# # # # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # # # #                 run_id = item 
# # # # #                 metrics_data = None
                
# # # # #                 if has_topk_folder:
# # # # #                     json_name = f"{run_id}_metrics.json"
# # # # #                     json_path = os.path.join(topk_results_path, json_name)
# # # # #                     if os.path.exists(json_path):
# # # # #                         try:
# # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # #                         except: pass
                
# # # # #                 if metrics_data is None:
# # # # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # # # #                     if os.path.exists(json_path):
# # # # #                         try:
# # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # #                         except: pass
                
# # # # #                 if metrics_data:
# # # # #                     entry = {
# # # # #                         'model': model_name,
# # # # #                         'config_id': params['config_id'],
# # # # #                         'loss': params['loss'],
# # # # #                         'selection_metric': params['metric'],
# # # # #                         'run': run_id,
# # # # #                         'data': metrics_data
# # # # #                     }
# # # # #                     experiments.append(entry)
                    
# # # # #     return experiments

# # # # # def get_k_metric(data, base_metric, k):
# # # # #     keys_to_try = [
# # # # #         f"{base_metric}_at_{k}_pct",
# # # # #         f"{base_metric}_at_{k:.1f}_pct",
# # # # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # # # #     ]
# # # # #     for key in keys_to_try:
# # # # #         if key in data: return data[key]
# # # # #     return None

# # # # # # ==============================================================================
# # # # # # ANALYSIS FUNCTIONS
# # # # # # ==============================================================================

# # # # # def analyze_part_a(experiments, logger):
# # # # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # # # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
# # # # #     logger.log("Values shown as: **mean ± std**\n")
    
# # # # #     groups = collections.defaultdict(list)
# # # # #     for exp in experiments:
# # # # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # # # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # # # #         groups[key].append(exp)
    
# # # # #     sorted_keys = sorted(groups.keys())

# # # # #     for metric_type in METRICS_OF_INTEREST:
# # # # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # # # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # # # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # # # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # # # #         separator = "".join(sep_cols)
        
# # # # #         logger.log(header)
# # # # #         logger.log(separator)
        
# # # # #         for key in sorted_keys:
# # # # #             model, loss, sel_metric = key
# # # # #             exps = groups[key]
            
# # # # #             # Group by config_id to get all runs of the same configuration
# # # # #             config_groups = collections.defaultdict(list)
# # # # #             for exp in exps:
# # # # #                 config_groups[exp['config_id']].append(exp)
            
# # # # #             # Find best configuration based on mean validation metric
# # # # #             best_config_id = None
# # # # #             best_mean_val = -1
# # # # #             for config_id, config_exps in config_groups.items():
# # # # #                 val_scores = [e['data'].get('best_validation_metric', -1) for e in config_exps]
# # # # #                 mean_val = np.mean(val_scores)
# # # # #                 if mean_val > best_mean_val:
# # # # #                     best_mean_val = mean_val
# # # # #                     best_config_id = config_id
            
# # # # #             # Get all runs of the best configuration
# # # # #             best_runs = config_groups[best_config_id]
            
# # # # #             # Compute mean and std for each metric
# # # # #             val_scores = [e['data'].get('best_validation_metric', -1) for e in best_runs]
# # # # #             roc_scores = [e['data'].get('auc_roc', 0) for e in best_runs]
# # # # #             pr_scores = [e['data'].get('auc_pr', 0) for e in best_runs]
            
# # # # #             val_mean, val_std = np.mean(val_scores), np.std(val_scores)
# # # # #             roc_mean, roc_std = np.mean(roc_scores), np.std(roc_scores)
# # # # #             pr_mean, pr_std = np.mean(pr_scores), np.std(pr_scores)
            
# # # # #             # Compute mean and std for each K value
# # # # #             k_strings = []
# # # # #             for k in K_VALUES:
# # # # #                 k_values = []
# # # # #                 for exp in best_runs:
# # # # #                     val_k = get_k_metric(exp['data'], metric_type, k)
# # # # #                     if val_k is not None:
# # # # #                         k_values.append(val_k)
                
# # # # #                 if k_values:
# # # # #                     k_mean = np.mean(k_values)
# # # # #                     k_std = np.std(k_values)
# # # # #                     val_str = f"{k_mean:.4f}±{k_std:.4f}"
# # # # #                 else:
# # # # #                     val_str = "-"
# # # # #                 k_strings.append(val_str)
            
# # # # #             k_row_str = " | ".join(k_strings)
            
# # # # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val_mean:.4f}±{val_std:.4f} | {roc_mean:.4f}±{roc_std:.4f} | {pr_mean:.4f}±{pr_std:.4f} | {k_row_str} |")
        
# # # # #         logger.log("\n")

# # # # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # # # #     paired_data = collections.defaultdict(dict)
# # # # #     for exp in experiments:
# # # # #         if filter_loss and exp['loss'] != filter_loss: continue
# # # # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # # # #         paired_data[uid][exp['selection_metric']] = exp
        
# # # # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # # # #     pair_count = 0
    
# # # # #     for uid, metrics in paired_data.items():
# # # # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # # # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # # # #             pair_count += 1
# # # # #             for m in METRICS_OF_INTEREST:
# # # # #                 for k in K_VALUES:
# # # # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # # # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # # # #                     if v_pr is not None and v_roc is not None:
# # # # #                         results[m][k].append((v_pr, v_roc))

# # # # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # # # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
    
# # # # #     # Updated Header to include Mean Ratio and % Gain
# # # # #     logger.log("| Metric | Top-K | Mean Ratio (Volatile) | Agg Ratio (Total) | Median Ratio (Typical) | % Gain (Robust) | Win Rate | P-Value |")
# # # # #     logger.log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
# # # # #     for m in METRICS_OF_INTEREST:
# # # # #         first = True
# # # # #         for k in K_VALUES:
# # # # #             pairs = results[m][k]
# # # # #             if not pairs: continue
# # # # #             arr_pr = np.array([p[0] for p in pairs])
# # # # #             arr_roc = np.array([p[1] for p in pairs])
            
# # # # #             # 1. Mean Ratio (Sensitive to outliers)
# # # # #             individual_ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
# # # # #             mean_ratio = np.mean(individual_ratios)
            
# # # # #             # 2. Aggregate Ratio (Robust to individual 0s)
# # # # #             agg_ratio = np.sum(arr_pr) / (np.sum(arr_roc) + 1e-9)
            
# # # # #             # 3. Median Ratio (Typical performance)
# # # # #             median_ratio = np.median(individual_ratios)
            
# # # # #             # 4. % Gain (Relative Improvement based on Median)
# # # # #             pct_gain = (median_ratio - 1) * 100
            
# # # # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # # # #             try: 
# # # # #                 diff = arr_pr - arr_roc
# # # # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # # # #             except: p_val = 1.0
            
# # # # #             # Formatting strings
# # # # #             label = f"**{m.upper()}**" if first else ""
# # # # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            
# # # # #             # Handle exploding Mean Ratio display
# # # # #             if mean_ratio > 100: mean_str = f"{mean_ratio:.1e}x"
# # # # #             else: mean_str = f"{mean_ratio:.3f}x"
                
# # # # #             agg_str = f"**{agg_ratio:.3f}x**" if agg_ratio > 1.0 else f"{agg_ratio:.3f}x"
# # # # #             med_str = f"**{median_ratio:.3f}x**" if median_ratio > 1.0 else f"{median_ratio:.3f}x"
            
# # # # #             # Gain string with +/- sign
# # # # #             gain_str = f"+{pct_gain:.1f}%" if pct_gain > 0 else f"{pct_gain:.1f}%"
# # # # #             if pct_gain > 0: gain_str = f"**{gain_str}**"
            
# # # # #             logger.log(f"| {label} | {k}% | {mean_str} | {agg_str} | {med_str} | {gain_str} | {win:.1f}% | {p_str} |")
# # # # #             first = False
# # # # #         logger.log("| | | | | | | | |")

# # # # # def main():
# # # # #     parser = argparse.ArgumentParser()
# # # # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # # # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # # # #     parser.add_argument("--model", type=str, default=None)
# # # # #     parser.add_argument("--filter", type=str, default=None)
# # # # #     args = parser.parse_args()
    
# # # # #     output_filename = f"analysis_report_{args.dataset}"
# # # # #     if args.model: output_filename += f"_{args.model}"
# # # # #     if args.filter: output_filename += f"_{args.filter}"
# # # # #     output_filename += ".md"
    
# # # # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # # # #     logger = ReportLogger(out_path)
    
# # # # #     header = f"# Analysis Report for {args.dataset}"
# # # # #     if args.model: header += f" ({args.model})"
# # # # #     if args.filter: header += f" [Filter: {args.filter}]"
# # # # #     logger.log(header)
    
# # # # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # # # #     if not exps: 
# # # # #         print("No results found.")
# # # # #         return
    
# # # # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # # # #     analyze_part_a(exps, logger)
# # # # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # # # #     perform_paired_analysis(exps, logger, None)
# # # # #     perform_paired_analysis(exps, logger, 'Standard')
# # # # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # # # #     logger.save()

# # # # # if __name__ == "__main__":
# # # # #     main()







# # # # # # import os
# # # # # # import json
# # # # # # import argparse
# # # # # # import numpy as np
# # # # # # from scipy.stats import wilcoxon
# # # # # # import collections
# # # # # # import warnings
# # # # # # import sys

# # # # # # warnings.filterwarnings("ignore")

# # # # # # # ==============================================================================
# # # # # # # CONFIGURATION
# # # # # # # ==============================================================================

# # # # # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # # # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # # # # ==============================================================================
# # # # # # # OUTPUT HELPER
# # # # # # # ==============================================================================

# # # # # # class ReportLogger:
# # # # # #     def __init__(self, filepath):
# # # # # #         self.filepath = filepath
# # # # # #         self.content = []
        
# # # # # #     def log(self, message=""):
# # # # # #         print(message)
# # # # # #         self.content.append(message)
        
# # # # # #     def save(self):
# # # # # #         with open(self.filepath, 'w') as f:
# # # # # #             f.write("\n".join(self.content))
# # # # # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # # # # ==============================================================================
# # # # # # # DATA LOADING
# # # # # # # ==============================================================================

# # # # # # def parse_folder_name(folder_name):
# # # # # #     """Parses folder string to extract parameters."""
# # # # # #     parts = folder_name.split('_')
# # # # # #     params = {}
# # # # # #     params['loss'] = 'Standard'
# # # # # #     params['metric'] = 'rocauc'
    
# # # # # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # # # # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # # # # #     for part in parts:
# # # # # #         if part.startswith('metric'):
# # # # # #             raw_val = part.replace('metric', '')
# # # # # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # # # # #             params['metric'] = raw_val
    
# # # # # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # # # # #     config_parts.sort()
# # # # # #     params['config_id'] = "_".join(config_parts)
# # # # # #     return params

# # # # # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # # # # #     experiments = []
# # # # # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # # # # #     if not os.path.exists(dataset_dir):
# # # # # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # # # # #         return []

# # # # # #     for model_name in os.listdir(dataset_dir):
# # # # # #         if target_model and model_name.lower() != target_model.lower():
# # # # # #             continue

# # # # # #         model_path = os.path.join(dataset_dir, model_name)
# # # # # #         if not os.path.isdir(model_path): continue
        
# # # # # #         for folder_name in os.listdir(model_path):
# # # # # #             folder_path = os.path.join(model_path, folder_name)
# # # # # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # # # # #                 continue
            
# # # # # #             if filter_str and filter_str not in folder_name:
# # # # # #                 continue
            
# # # # # #             params = parse_folder_name(folder_name)
# # # # # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # # # # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # # # # #             for item in os.listdir(folder_path):
# # # # # #                 if not item.startswith('run_'): continue
# # # # # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # # # # #                 run_id = item 
# # # # # #                 metrics_data = None
                
# # # # # #                 if has_topk_folder:
# # # # # #                     json_name = f"{run_id}_metrics.json"
# # # # # #                     json_path = os.path.join(topk_results_path, json_name)
# # # # # #                     if os.path.exists(json_path):
# # # # # #                         try:
# # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # #                         except: pass
                
# # # # # #                 if metrics_data is None:
# # # # # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # # # # #                     if os.path.exists(json_path):
# # # # # #                         try:
# # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # #                         except: pass
                
# # # # # #                 if metrics_data:
# # # # # #                     entry = {
# # # # # #                         'model': model_name,
# # # # # #                         'config_id': params['config_id'],
# # # # # #                         'loss': params['loss'],
# # # # # #                         'selection_metric': params['metric'],
# # # # # #                         'run': run_id,
# # # # # #                         'data': metrics_data
# # # # # #                     }
# # # # # #                     experiments.append(entry)
                    
# # # # # #     return experiments

# # # # # # def get_k_metric(data, base_metric, k):
# # # # # #     keys_to_try = [
# # # # # #         f"{base_metric}_at_{k}_pct",
# # # # # #         f"{base_metric}_at_{k:.1f}_pct",
# # # # # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # # # # #     ]
# # # # # #     for key in keys_to_try:
# # # # # #         if key in data: return data[key]
# # # # # #     return None

# # # # # # # ==============================================================================
# # # # # # # ANALYSIS FUNCTIONS
# # # # # # # ==============================================================================

# # # # # # def analyze_part_a(experiments, logger):
# # # # # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # # # # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# # # # # #     groups = collections.defaultdict(list)
# # # # # #     for exp in experiments:
# # # # # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # # # # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # # # # #         groups[key].append(exp)
    
# # # # # #     sorted_keys = sorted(groups.keys())

# # # # # #     for metric_type in METRICS_OF_INTEREST:
# # # # # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # # # # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # # # # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # # # # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # # # # #         separator = "".join(sep_cols)
        
# # # # # #         logger.log(header)
# # # # # #         logger.log(separator)
        
# # # # # #         for key in sorted_keys:
# # # # # #             model, loss, sel_metric = key
# # # # # #             exps = groups[key]
            
# # # # # #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# # # # # #             d = best_exp['data']
            
# # # # # #             val = d.get('best_validation_metric', -1)
# # # # # #             roc = d.get('auc_roc', 0)
# # # # # #             pr = d.get('auc_pr', 0)
            
# # # # # #             k_strings = []
# # # # # #             for k in K_VALUES:
# # # # # #                 val_k = get_k_metric(d, metric_type, k)
# # # # # #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# # # # # #                 k_strings.append(val_str)
            
# # # # # #             k_row_str = " | ".join(k_strings)
            
# # # # # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# # # # # #         logger.log("\n")

# # # # # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # # # # #     paired_data = collections.defaultdict(dict)
# # # # # #     for exp in experiments:
# # # # # #         if filter_loss and exp['loss'] != filter_loss: continue
# # # # # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # # # # #         paired_data[uid][exp['selection_metric']] = exp
        
# # # # # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # # # # #     pair_count = 0
    
# # # # # #     for uid, metrics in paired_data.items():
# # # # # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # # # # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # # # # #             pair_count += 1
# # # # # #             for m in METRICS_OF_INTEREST:
# # # # # #                 for k in K_VALUES:
# # # # # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # # # # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # # # # #                     if v_pr is not None and v_roc is not None:
# # # # # #                         results[m][k].append((v_pr, v_roc))

# # # # # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # # # # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
    
# # # # # #     # Updated Header to include Mean Ratio and % Gain
# # # # # #     logger.log("| Metric | Top-K | Mean Ratio (Volatile) | Agg Ratio (Total) | Median Ratio (Typical) | % Gain (Robust) | Win Rate | P-Value |")
# # # # # #     logger.log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
# # # # # #     for m in METRICS_OF_INTEREST:
# # # # # #         first = True
# # # # # #         for k in K_VALUES:
# # # # # #             pairs = results[m][k]
# # # # # #             if not pairs: continue
# # # # # #             arr_pr = np.array([p[0] for p in pairs])
# # # # # #             arr_roc = np.array([p[1] for p in pairs])
            
# # # # # #             # 1. Mean Ratio (Sensitive to outliers)
# # # # # #             individual_ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
# # # # # #             mean_ratio = np.mean(individual_ratios)
            
# # # # # #             # 2. Aggregate Ratio (Robust to individual 0s)
# # # # # #             agg_ratio = np.sum(arr_pr) / (np.sum(arr_roc) + 1e-9)
            
# # # # # #             # 3. Median Ratio (Typical performance)
# # # # # #             median_ratio = np.median(individual_ratios)
            
# # # # # #             # 4. % Gain (Relative Improvement based on Median)
# # # # # #             pct_gain = (median_ratio - 1) * 100
            
# # # # # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # # # # #             try: 
# # # # # #                 diff = arr_pr - arr_roc
# # # # # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # # # # #             except: p_val = 1.0
            
# # # # # #             # Formatting strings
# # # # # #             label = f"**{m.upper()}**" if first else ""
# # # # # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            
# # # # # #             # Handle exploding Mean Ratio display
# # # # # #             if mean_ratio > 100: mean_str = f"{mean_ratio:.1e}x"
# # # # # #             else: mean_str = f"{mean_ratio:.3f}x"
                
# # # # # #             agg_str = f"**{agg_ratio:.3f}x**" if agg_ratio > 1.0 else f"{agg_ratio:.3f}x"
# # # # # #             med_str = f"**{median_ratio:.3f}x**" if median_ratio > 1.0 else f"{median_ratio:.3f}x"
            
# # # # # #             # Gain string with +/- sign
# # # # # #             gain_str = f"+{pct_gain:.1f}%" if pct_gain > 0 else f"{pct_gain:.1f}%"
# # # # # #             if pct_gain > 0: gain_str = f"**{gain_str}**"
            
# # # # # #             logger.log(f"| {label} | {k}% | {mean_str} | {agg_str} | {med_str} | {gain_str} | {win:.1f}% | {p_str} |")
# # # # # #             first = False
# # # # # #         logger.log("| | | | | | | | |")

# # # # # # def main():
# # # # # #     parser = argparse.ArgumentParser()
# # # # # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # # # # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # # # # #     parser.add_argument("--model", type=str, default=None)
# # # # # #     parser.add_argument("--filter", type=str, default=None)
# # # # # #     args = parser.parse_args()
    
# # # # # #     output_filename = f"analysis_report_{args.dataset}"
# # # # # #     if args.model: output_filename += f"_{args.model}"
# # # # # #     if args.filter: output_filename += f"_{args.filter}"
# # # # # #     output_filename += ".md"
    
# # # # # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # # # # #     logger = ReportLogger(out_path)
    
# # # # # #     header = f"# Analysis Report for {args.dataset}"
# # # # # #     if args.model: header += f" ({args.model})"
# # # # # #     if args.filter: header += f" [Filter: {args.filter}]"
# # # # # #     logger.log(header)
    
# # # # # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # # # # #     if not exps: 
# # # # # #         print("No results found.")
# # # # # #         return
    
# # # # # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # # # # #     analyze_part_a(exps, logger)
# # # # # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # # # # #     perform_paired_analysis(exps, logger, None)
# # # # # #     perform_paired_analysis(exps, logger, 'Standard')
# # # # # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # # # # #     logger.save()

# # # # # # if __name__ == "__main__":
# # # # # #     main()












# # # # # # import os
# # # # # # import json
# # # # # # import argparse
# # # # # # import numpy as np
# # # # # # from scipy.stats import wilcoxon
# # # # # # import collections
# # # # # # import warnings
# # # # # # import sys

# # # # # # warnings.filterwarnings("ignore")

# # # # # # # ==============================================================================
# # # # # # # CONFIGURATION
# # # # # # # ==============================================================================

# # # # # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # # # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # # # # ==============================================================================
# # # # # # # OUTPUT HELPER
# # # # # # # ==============================================================================

# # # # # # class ReportLogger:
# # # # # #     def __init__(self, filepath):
# # # # # #         self.filepath = filepath
# # # # # #         self.content = []
        
# # # # # #     def log(self, message=""):
# # # # # #         print(message)
# # # # # #         self.content.append(message)
        
# # # # # #     def save(self):
# # # # # #         with open(self.filepath, 'w') as f:
# # # # # #             f.write("\n".join(self.content))
# # # # # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # # # # ==============================================================================
# # # # # # # DATA LOADING
# # # # # # # ==============================================================================

# # # # # # def parse_folder_name(folder_name):
# # # # # #     parts = folder_name.split('_')
# # # # # #     params = {}
# # # # # #     params['loss'] = 'Standard'
# # # # # #     params['metric'] = 'rocauc'
    
# # # # # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # # # # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # # # # #     for part in parts:
# # # # # #         if part.startswith('metric'):
# # # # # #             raw_val = part.replace('metric', '')
# # # # # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # # # # #             params['metric'] = raw_val
    
# # # # # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # # # # #     config_parts.sort()
# # # # # #     params['config_id'] = "_".join(config_parts)
# # # # # #     return params

# # # # # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # # # # #     experiments = []
# # # # # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # # # # #     if not os.path.exists(dataset_dir):
# # # # # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # # # # #         return []

# # # # # #     for model_name in os.listdir(dataset_dir):
# # # # # #         if target_model and model_name.lower() != target_model.lower():
# # # # # #             continue

# # # # # #         model_path = os.path.join(dataset_dir, model_name)
# # # # # #         if not os.path.isdir(model_path): continue
        
# # # # # #         for folder_name in os.listdir(model_path):
# # # # # #             folder_path = os.path.join(model_path, folder_name)
# # # # # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # # # # #                 continue
            
# # # # # #             if filter_str and filter_str not in folder_name:
# # # # # #                 continue
            
# # # # # #             params = parse_folder_name(folder_name)
# # # # # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # # # # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # # # # #             for item in os.listdir(folder_path):
# # # # # #                 if not item.startswith('run_'): continue
# # # # # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # # # # #                 run_id = item 
# # # # # #                 metrics_data = None
                
# # # # # #                 if has_topk_folder:
# # # # # #                     json_name = f"{run_id}_metrics.json"
# # # # # #                     json_path = os.path.join(topk_results_path, json_name)
# # # # # #                     if os.path.exists(json_path):
# # # # # #                         try:
# # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # #                         except: pass
                
# # # # # #                 if metrics_data is None:
# # # # # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # # # # #                     if os.path.exists(json_path):
# # # # # #                         try:
# # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # #                         except: pass
                
# # # # # #                 if metrics_data:
# # # # # #                     entry = {
# # # # # #                         'model': model_name,
# # # # # #                         'config_id': params['config_id'],
# # # # # #                         'loss': params['loss'],
# # # # # #                         'selection_metric': params['metric'],
# # # # # #                         'run': run_id,
# # # # # #                         'data': metrics_data
# # # # # #                     }
# # # # # #                     experiments.append(entry)
                    
# # # # # #     return experiments

# # # # # # def get_k_metric(data, base_metric, k):
# # # # # #     keys_to_try = [
# # # # # #         f"{base_metric}_at_{k}_pct",
# # # # # #         f"{base_metric}_at_{k:.1f}_pct",
# # # # # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # # # # #     ]
# # # # # #     for key in keys_to_try:
# # # # # #         if key in data: return data[key]
# # # # # #     return None

# # # # # # # ==============================================================================
# # # # # # # ANALYSIS FUNCTIONS
# # # # # # # ==============================================================================

# # # # # # def analyze_part_a(experiments, logger):
# # # # # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # # # # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# # # # # #     groups = collections.defaultdict(list)
# # # # # #     for exp in experiments:
# # # # # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # # # # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # # # # #         groups[key].append(exp)
    
# # # # # #     sorted_keys = sorted(groups.keys())

# # # # # #     for metric_type in METRICS_OF_INTEREST:
# # # # # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # # # # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # # # # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # # # # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # # # # #         separator = "".join(sep_cols)
        
# # # # # #         logger.log(header)
# # # # # #         logger.log(separator)
        
# # # # # #         for key in sorted_keys:
# # # # # #             model, loss, sel_metric = key
# # # # # #             exps = groups[key]
            
# # # # # #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# # # # # #             d = best_exp['data']
            
# # # # # #             val = d.get('best_validation_metric', -1)
# # # # # #             roc = d.get('auc_roc', 0)
# # # # # #             pr = d.get('auc_pr', 0)
            
# # # # # #             k_strings = []
# # # # # #             for k in K_VALUES:
# # # # # #                 val_k = get_k_metric(d, metric_type, k)
# # # # # #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# # # # # #                 k_strings.append(val_str)
            
# # # # # #             k_row_str = " | ".join(k_strings)
            
# # # # # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# # # # # #         logger.log("\n")

# # # # # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # # # # #     paired_data = collections.defaultdict(dict)
# # # # # #     for exp in experiments:
# # # # # #         if filter_loss and exp['loss'] != filter_loss: continue
# # # # # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # # # # #         paired_data[uid][exp['selection_metric']] = exp
        
# # # # # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # # # # #     pair_count = 0
    
# # # # # #     for uid, metrics in paired_data.items():
# # # # # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # # # # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # # # # #             pair_count += 1
# # # # # #             for m in METRICS_OF_INTEREST:
# # # # # #                 for k in K_VALUES:
# # # # # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # # # # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # # # # #                     if v_pr is not None and v_roc is not None:
# # # # # #                         results[m][k].append((v_pr, v_roc))

# # # # # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # # # # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
# # # # # #     # Updated Header: Replaced "Mean Ratio" with "Agg Ratio" and "Median Ratio"
# # # # # #     logger.log("| Metric | Top-K | Agg Ratio (Total/Total) | Median Ratio (Run-wise) | Win Rate (PR > ROC) | P-Value |")
# # # # # #     logger.log("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
# # # # # #     for m in METRICS_OF_INTEREST:
# # # # # #         first = True
# # # # # #         for k in K_VALUES:
# # # # # #             pairs = results[m][k]
# # # # # #             if not pairs: continue
# # # # # #             arr_pr = np.array([p[0] for p in pairs])
# # # # # #             arr_roc = np.array([p[1] for p in pairs])
            
# # # # # #             # 1. Aggregate Ratio (Robust to individual 0s)
# # # # # #             # Sum of PR scores / Sum of ROC scores
# # # # # #             agg_ratio = np.sum(arr_pr) / (np.sum(arr_roc) + 1e-9)
            
# # # # # #             # 2. Median Ratio (Robust to outliers)
# # # # # #             # We calculate individual ratios first
# # # # # #             individual_ratios = (arr_pr + 1e-9) / (arr_roc + 1e-9)
# # # # # #             median_ratio = np.median(individual_ratios)
            
# # # # # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # # # # #             try: 
# # # # # #                 diff = arr_pr - arr_roc
# # # # # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # # # # #             except: p_val = 1.0
            
# # # # # #             label = f"**{m.upper()}**" if first else ""
# # # # # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
            
# # # # # #             agg_str = f"**{agg_ratio:.3f}x**" if agg_ratio > 1.0 else f"{agg_ratio:.3f}x"
# # # # # #             med_str = f"**{median_ratio:.3f}x**" if median_ratio > 1.0 else f"{median_ratio:.3f}x"
            
# # # # # #             logger.log(f"| {label} | {k}% | {agg_str} | {med_str} | {win:.1f}% | {p_str} |")
# # # # # #             first = False
# # # # # #         logger.log("| | | | | | |")

# # # # # # def main():
# # # # # #     parser = argparse.ArgumentParser()
# # # # # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # # # # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # # # # #     parser.add_argument("--model", type=str, default=None)
# # # # # #     parser.add_argument("--filter", type=str, default=None)
# # # # # #     args = parser.parse_args()
    
# # # # # #     output_filename = f"analysis_report_{args.dataset}"
# # # # # #     if args.model: output_filename += f"_{args.model}"
# # # # # #     if args.filter: output_filename += f"_{args.filter}"
# # # # # #     output_filename += ".md"
    
# # # # # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # # # # #     logger = ReportLogger(out_path)
    
# # # # # #     header = f"# Analysis Report for {args.dataset}"
# # # # # #     if args.model: header += f" ({args.model})"
# # # # # #     if args.filter: header += f" [Filter: {args.filter}]"
# # # # # #     logger.log(header)
    
# # # # # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # # # # #     if not exps: 
# # # # # #         print("No results found.")
# # # # # #         return
    
# # # # # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # # # # #     analyze_part_a(exps, logger)
# # # # # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # # # # #     perform_paired_analysis(exps, logger, None)
# # # # # #     perform_paired_analysis(exps, logger, 'Standard')
# # # # # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # # # # #     logger.save()

# # # # # # if __name__ == "__main__":
# # # # # #     main()




# # # # # # # import os
# # # # # # # import json
# # # # # # # import argparse
# # # # # # # import numpy as np
# # # # # # # from scipy.stats import wilcoxon
# # # # # # # import collections
# # # # # # # import warnings
# # # # # # # import sys

# # # # # # # warnings.filterwarnings("ignore")

# # # # # # # # ==============================================================================
# # # # # # # # CONFIGURATION
# # # # # # # # ==============================================================================

# # # # # # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # # # # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # # # # # ==============================================================================
# # # # # # # # OUTPUT HELPER
# # # # # # # # ==============================================================================

# # # # # # # class ReportLogger:
# # # # # # #     def __init__(self, filepath):
# # # # # # #         self.filepath = filepath
# # # # # # #         self.content = []
        
# # # # # # #     def log(self, message=""):
# # # # # # #         print(message)
# # # # # # #         self.content.append(message)
        
# # # # # # #     def save(self):
# # # # # # #         with open(self.filepath, 'w') as f:
# # # # # # #             f.write("\n".join(self.content))
# # # # # # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # # # # # ==============================================================================
# # # # # # # # DATA LOADING
# # # # # # # # ==============================================================================

# # # # # # # def parse_folder_name(folder_name):
# # # # # # #     """Parses folder string to extract parameters."""
# # # # # # #     parts = folder_name.split('_')
# # # # # # #     params = {}
# # # # # # #     params['loss'] = 'Standard'
# # # # # # #     params['metric'] = 'rocauc'
    
# # # # # # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # # # # # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # # # # # #     for part in parts:
# # # # # # #         if part.startswith('metric'):
# # # # # # #             raw_val = part.replace('metric', '')
# # # # # # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # # # # # #             params['metric'] = raw_val
    
# # # # # # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # # # # # #     config_parts.sort()
# # # # # # #     params['config_id'] = "_".join(config_parts)
# # # # # # #     return params

# # # # # # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # # # # # #     experiments = []
# # # # # # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # # # # # #     if not os.path.exists(dataset_dir):
# # # # # # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # # # # # #         return []

# # # # # # #     for model_name in os.listdir(dataset_dir):
# # # # # # #         # Model Filter
# # # # # # #         if target_model and model_name.lower() != target_model.lower():
# # # # # # #             continue

# # # # # # #         model_path = os.path.join(dataset_dir, model_name)
# # # # # # #         if not os.path.isdir(model_path): continue
        
# # # # # # #         for folder_name in os.listdir(model_path):
# # # # # # #             # 1. Skip system folders
# # # # # # #             folder_path = os.path.join(model_path, folder_name)
# # # # # # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # # # # # #                 continue
            
# # # # # # #             # 2. String Filter (e.g. 'lr-0.0001')
# # # # # # #             if filter_str and filter_str not in folder_name:
# # # # # # #                 continue
            
# # # # # # #             params = parse_folder_name(folder_name)
# # # # # # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # # # # # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # # # # # #             for item in os.listdir(folder_path):
# # # # # # #                 if not item.startswith('run_'): continue
# # # # # # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # # # # # #                 run_id = item 
# # # # # # #                 metrics_data = None
                
# # # # # # #                 if has_topk_folder:
# # # # # # #                     json_name = f"{run_id}_metrics.json"
# # # # # # #                     json_path = os.path.join(topk_results_path, json_name)
# # # # # # #                     if os.path.exists(json_path):
# # # # # # #                         try:
# # # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # # #                         except: pass
                
# # # # # # #                 if metrics_data is None:
# # # # # # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # # # # # #                     if os.path.exists(json_path):
# # # # # # #                         try:
# # # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # # #                         except: pass
                
# # # # # # #                 if metrics_data:
# # # # # # #                     entry = {
# # # # # # #                         'model': model_name,
# # # # # # #                         'config_id': params['config_id'],
# # # # # # #                         'loss': params['loss'],
# # # # # # #                         'selection_metric': params['metric'],
# # # # # # #                         'run': run_id,
# # # # # # #                         'data': metrics_data
# # # # # # #                     }
# # # # # # #                     experiments.append(entry)
                    
# # # # # # #     return experiments

# # # # # # # def get_k_metric(data, base_metric, k):
# # # # # # #     keys_to_try = [
# # # # # # #         f"{base_metric}_at_{k}_pct",
# # # # # # #         f"{base_metric}_at_{k:.1f}_pct",
# # # # # # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # # # # # #     ]
# # # # # # #     for key in keys_to_try:
# # # # # # #         if key in data: return data[key]
# # # # # # #     return None

# # # # # # # # ==============================================================================
# # # # # # # # ANALYSIS FUNCTIONS
# # # # # # # # ==============================================================================

# # # # # # # def analyze_part_a(experiments, logger):
# # # # # # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # # # # # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# # # # # # #     # 1. Group Experiments
# # # # # # #     groups = collections.defaultdict(list)
# # # # # # #     for exp in experiments:
# # # # # # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # # # # # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # # # # # #         groups[key].append(exp)
    
# # # # # # #     sorted_keys = sorted(groups.keys())

# # # # # # #     # 2. Iterate through each Metric type (NDCG, Precision, Recall)
# # # # # # #     for metric_type in METRICS_OF_INTEREST:
# # # # # # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # # # # # #         # Build Header
# # # # # # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # # # # # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # # # # # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # # # # # #         separator = "".join(sep_cols)
        
# # # # # # #         logger.log(header)
# # # # # # #         logger.log(separator)
        
# # # # # # #         for key in sorted_keys:
# # # # # # #             model, loss, sel_metric = key
# # # # # # #             exps = groups[key]
            
# # # # # # #             # Find Champion based on Validation Score
# # # # # # #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# # # # # # #             d = best_exp['data']
            
# # # # # # #             # Metadata
# # # # # # #             val = d.get('best_validation_metric', -1)
# # # # # # #             roc = d.get('auc_roc', 0)
# # # # # # #             pr = d.get('auc_pr', 0)
            
# # # # # # #             # Collect K values
# # # # # # #             k_strings = []
# # # # # # #             for k in K_VALUES:
# # # # # # #                 val_k = get_k_metric(d, metric_type, k)
# # # # # # #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# # # # # # #                 k_strings.append(val_str)
            
# # # # # # #             k_row_str = " | ".join(k_strings)
            
# # # # # # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# # # # # # #         logger.log("\n")

# # # # # # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # # # # # #     paired_data = collections.defaultdict(dict)
# # # # # # #     for exp in experiments:
# # # # # # #         if filter_loss and exp['loss'] != filter_loss: continue
# # # # # # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # # # # # #         paired_data[uid][exp['selection_metric']] = exp
        
# # # # # # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # # # # # #     pair_count = 0
    
# # # # # # #     for uid, metrics in paired_data.items():
# # # # # # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # # # # # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # # # # # #             pair_count += 1
# # # # # # #             for m in METRICS_OF_INTEREST:
# # # # # # #                 for k in K_VALUES:
# # # # # # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # # # # # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # # # # # #                     if v_pr is not None and v_roc is not None:
# # # # # # #                         results[m][k].append((v_pr, v_roc))

# # # # # # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # # # # # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
# # # # # # #     logger.log("| Metric | Top-K | Mean Ratio (PR/ROC) | Win Rate (PR > ROC) | P-Value |")
# # # # # # #     logger.log("| :--- | :--- | :--- | :--- | :--- |")
    
# # # # # # #     for m in METRICS_OF_INTEREST:
# # # # # # #         first = True
# # # # # # #         for k in K_VALUES:
# # # # # # #             pairs = results[m][k]
# # # # # # #             if not pairs: continue
# # # # # # #             arr_pr = np.array([p[0] for p in pairs])
# # # # # # #             arr_roc = np.array([p[1] for p in pairs])
            
# # # # # # #             ratio = np.mean((arr_pr + 1e-9)/(arr_roc + 1e-9))
# # # # # # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # # # # # #             try: 
# # # # # # #                 diff = arr_pr - arr_roc
# # # # # # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # # # # # #             except: p_val = 1.0
            
# # # # # # #             label = f"**{m.upper()}**" if first else ""
# # # # # # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
# # # # # # #             r_str = f"**{ratio:.3f}x**" if ratio > 1.0 else f"{ratio:.3f}x"
# # # # # # #             logger.log(f"| {label} | {k}% | {r_str} | {win:.1f}% | {p_str} |")
# # # # # # #             first = False
# # # # # # #         logger.log("| | | | | |")

# # # # # # # def main():
# # # # # # #     parser = argparse.ArgumentParser()
# # # # # # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # # # # # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # # # # # #     parser.add_argument("--model", type=str, default=None, help="Filter by model (e.g. gcn)")
# # # # # # #     parser.add_argument("--filter", type=str, default=None, help="Filter by folder name string (e.g. lr-0.0001)")
# # # # # # #     args = parser.parse_args()
    
# # # # # # #     # Filename includes filter details
# # # # # # #     output_filename = f"analysis_report_{args.dataset}"
# # # # # # #     if args.model: output_filename += f"_{args.model}"
# # # # # # #     if args.filter: output_filename += f"_{args.filter}"
# # # # # # #     output_filename += ".md"
    
# # # # # # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # # # # # #     logger = ReportLogger(out_path)
    
# # # # # # #     header = f"# Analysis Report for {args.dataset}"
# # # # # # #     if args.model: header += f" ({args.model})"
# # # # # # #     if args.filter: header += f" [Filter: {args.filter}]"
# # # # # # #     logger.log(header)
    
# # # # # # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # # # # # #     if not exps: 
# # # # # # #         print("No results found matching your criteria.")
# # # # # # #         return
    
# # # # # # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # # # # # #     analyze_part_a(exps, logger)
# # # # # # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # # # # # #     perform_paired_analysis(exps, logger, None)
# # # # # # #     perform_paired_analysis(exps, logger, 'Standard')
# # # # # # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # # # # # #     logger.save()

# # # # # # # if __name__ == "__main__":
# # # # # # #     main()





# # # # # # # import os
# # # # # # # import json
# # # # # # # import argparse
# # # # # # # import numpy as np
# # # # # # # from scipy.stats import wilcoxon
# # # # # # # import collections
# # # # # # # import warnings
# # # # # # # import sys

# # # # # # # warnings.filterwarnings("ignore")

# # # # # # # # ==============================================================================
# # # # # # # # CONFIGURATION
# # # # # # # # ==============================================================================

# # # # # # # K_VALUES = [0.5, 1, 2, 3, 4, 5, 10]
# # # # # # # METRICS_OF_INTEREST = ['ndcg', 'precision', 'recall']

# # # # # # # # ==============================================================================
# # # # # # # # OUTPUT HELPER
# # # # # # # # ==============================================================================

# # # # # # # class ReportLogger:
# # # # # # #     def __init__(self, filepath):
# # # # # # #         self.filepath = filepath
# # # # # # #         self.content = []
        
# # # # # # #     def log(self, message=""):
# # # # # # #         print(message)
# # # # # # #         self.content.append(message)
        
# # # # # # #     def save(self):
# # # # # # #         with open(self.filepath, 'w') as f:
# # # # # # #             f.write("\n".join(self.content))
# # # # # # #         print(f"\n[Saved report to: {self.filepath}]")

# # # # # # # # ==============================================================================
# # # # # # # # DATA LOADING
# # # # # # # # ==============================================================================

# # # # # # # def parse_folder_name(folder_name):
# # # # # # #     """Parses folder string to extract parameters."""
# # # # # # #     parts = folder_name.split('_')
# # # # # # #     params = {}
# # # # # # #     params['loss'] = 'Standard'
# # # # # # #     params['metric'] = 'rocauc'
    
# # # # # # #     if 'class_weight' in parts: params['loss'] = 'Standard' 
# # # # # # #     elif any(p.startswith('focal') for p in parts): params['loss'] = 'Focal'
        
# # # # # # #     for part in parts:
# # # # # # #         if part.startswith('metric'):
# # # # # # #             raw_val = part.replace('metric', '')
# # # # # # #             if raw_val.startswith('-'): raw_val = raw_val[1:]
# # # # # # #             params['metric'] = raw_val
    
# # # # # # #     config_parts = [p for p in parts if not p.startswith('metric')]
# # # # # # #     config_parts.sort()
# # # # # # #     params['config_id'] = "_".join(config_parts)
# # # # # # #     return params

# # # # # # # def load_all_results(base_dir, dataset, target_model=None, filter_str=None):
# # # # # # #     experiments = []
# # # # # # #     dataset_dir = os.path.join(base_dir, dataset)
    
# # # # # # #     if not os.path.exists(dataset_dir):
# # # # # # #         print(f"Error: Dataset directory not found: {dataset_dir}")
# # # # # # #         return []

# # # # # # #     for model_name in os.listdir(dataset_dir):
# # # # # # #         # Model Filter
# # # # # # #         if target_model and model_name.lower() != target_model.lower():
# # # # # # #             continue

# # # # # # #         model_path = os.path.join(dataset_dir, model_name)
# # # # # # #         if not os.path.isdir(model_path): continue
        
# # # # # # #         for folder_name in os.listdir(model_path):
# # # # # # #             # 1. Skip system folders
# # # # # # #             folder_path = os.path.join(model_path, folder_name)
# # # # # # #             if not os.path.isdir(folder_path) or folder_name in ['summary', 'topk_percent', 'topk_results']:
# # # # # # #                 continue
            
# # # # # # #             # 2. String Filter (e.g. 'lr-0.0001')
# # # # # # #             if filter_str and filter_str not in folder_name:
# # # # # # #                 continue
            
# # # # # # #             params = parse_folder_name(folder_name)
# # # # # # #             topk_results_path = os.path.join(folder_path, 'topk_results')
# # # # # # #             has_topk_folder = os.path.exists(topk_results_path)
            
# # # # # # #             for item in os.listdir(folder_path):
# # # # # # #                 if not item.startswith('run_'): continue
# # # # # # #                 if not os.path.isdir(os.path.join(folder_path, item)): continue
                
# # # # # # #                 run_id = item 
# # # # # # #                 metrics_data = None
                
# # # # # # #                 if has_topk_folder:
# # # # # # #                     json_name = f"{run_id}_metrics.json"
# # # # # # #                     json_path = os.path.join(topk_results_path, json_name)
# # # # # # #                     if os.path.exists(json_path):
# # # # # # #                         try:
# # # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # # #                         except: pass
                
# # # # # # #                 if metrics_data is None:
# # # # # # #                     json_path = os.path.join(folder_path, run_id, 'metrics.json')
# # # # # # #                     if os.path.exists(json_path):
# # # # # # #                         try:
# # # # # # #                             with open(json_path, 'r') as f: metrics_data = json.load(f)
# # # # # # #                         except: pass
                
# # # # # # #                 if metrics_data:
# # # # # # #                     entry = {
# # # # # # #                         'model': model_name,
# # # # # # #                         'config_id': params['config_id'],
# # # # # # #                         'loss': params['loss'],
# # # # # # #                         'selection_metric': params['metric'],
# # # # # # #                         'run': run_id,
# # # # # # #                         'data': metrics_data
# # # # # # #                     }
# # # # # # #                     experiments.append(entry)
                    
# # # # # # #     return experiments

# # # # # # # def get_k_metric(data, base_metric, k):
# # # # # # #     keys_to_try = [
# # # # # # #         f"{base_metric}_at_{k}_pct",
# # # # # # #         f"{base_metric}_at_{k:.1f}_pct",
# # # # # # #         f"{base_metric}_at_{int(k)}_pct" if k == int(k) else "IGNORE"
# # # # # # #     ]
# # # # # # #     for key in keys_to_try:
# # # # # # #         if key in data: return data[key]
# # # # # # #     return None

# # # # # # # # ==============================================================================
# # # # # # # # ANALYSIS FUNCTIONS
# # # # # # # # ==============================================================================

# # # # # # # def analyze_part_a(experiments, logger):
# # # # # # #     logger.log(f"\n## Part A: Best Model Performance (Per GNN)\n")
# # # # # # #     logger.log("Comparison of the best configuration found for each Model/Loss/Metric combination across all K thresholds.\n")
    
# # # # # # #     # 1. Group Experiments
# # # # # # #     groups = collections.defaultdict(list)
# # # # # # #     for exp in experiments:
# # # # # # #         if exp['selection_metric'].lower() not in ['rocauc', 'prauc']: continue
# # # # # # #         key = (exp['model'], exp['loss'], exp['selection_metric'])
# # # # # # #         groups[key].append(exp)
    
# # # # # # #     sorted_keys = sorted(groups.keys())

# # # # # # #     # 2. Iterate through each Metric type (NDCG, Precision, Recall)
# # # # # # #     for metric_type in METRICS_OF_INTEREST:
# # # # # # #         logger.log(f"### Best Models: {metric_type.upper()} Evolution")
        
# # # # # # #         # Build Header
# # # # # # #         k_headers = " | ".join([f"@{k}%" for k in K_VALUES])
# # # # # # #         header = f"| Model | Loss | Selection | Val Score | Test ROC | Test PR | {k_headers} |"
# # # # # # #         sep_cols = ["| :---"] * 6 + [" :--- |"] * len(K_VALUES)
# # # # # # #         separator = "".join(sep_cols)
        
# # # # # # #         logger.log(header)
# # # # # # #         logger.log(separator)
        
# # # # # # #         for key in sorted_keys:
# # # # # # #             model, loss, sel_metric = key
# # # # # # #             exps = groups[key]
            
# # # # # # #             # Find Champion based on Validation Score
# # # # # # #             best_exp = max(exps, key=lambda x: x['data'].get('best_validation_metric', -1))
# # # # # # #             d = best_exp['data']
            
# # # # # # #             # Metadata
# # # # # # #             val = d.get('best_validation_metric', -1)
# # # # # # #             roc = d.get('auc_roc', 0)
# # # # # # #             pr = d.get('auc_pr', 0)
            
# # # # # # #             # Collect K values
# # # # # # #             k_strings = []
# # # # # # #             for k in K_VALUES:
# # # # # # #                 val_k = get_k_metric(d, metric_type, k)
# # # # # # #                 val_str = f"{val_k:.4f}" if val_k is not None else "-"
# # # # # # #                 k_strings.append(val_str)
            
# # # # # # #             k_row_str = " | ".join(k_strings)
            
# # # # # # #             logger.log(f"| **{model}** | {loss} | {sel_metric.upper()} | {val:.4f} | {roc:.4f} | {pr:.4f} | {k_row_str} |")
        
# # # # # # #         logger.log("\n")

# # # # # # # def perform_paired_analysis(experiments, logger, filter_loss=None):
# # # # # # #     paired_data = collections.defaultdict(dict)
# # # # # # #     for exp in experiments:
# # # # # # #         if filter_loss and exp['loss'] != filter_loss: continue
# # # # # # #         uid = (exp['model'], exp['config_id'], exp['run'])
# # # # # # #         paired_data[uid][exp['selection_metric']] = exp
        
# # # # # # #     results = collections.defaultdict(lambda: collections.defaultdict(list))
# # # # # # #     pair_count = 0
    
# # # # # # #     for uid, metrics in paired_data.items():
# # # # # # #         if 'prauc' in metrics and 'rocauc' in metrics:
# # # # # # #             if get_k_metric(metrics['prauc']['data'], 'ndcg', 1) is None: continue
            
# # # # # # #             pair_count += 1
# # # # # # #             for m in METRICS_OF_INTEREST:
# # # # # # #                 for k in K_VALUES:
# # # # # # #                     v_pr = get_k_metric(metrics['prauc']['data'], m, k)
# # # # # # #                     v_roc = get_k_metric(metrics['rocauc']['data'], m, k)
# # # # # # #                     if v_pr is not None and v_roc is not None:
# # # # # # #                         results[m][k].append((v_pr, v_roc))

# # # # # # #     logger.log(f"### Paired Analysis: {filter_loss if filter_loss else 'Global'}")
# # # # # # #     logger.log(f"*(Based on {pair_count} comparisons)*\n")
# # # # # # #     logger.log("| Metric | Top-K | Mean Ratio (PR/ROC) | Win Rate (PR > ROC) | P-Value |")
# # # # # # #     logger.log("| :--- | :--- | :--- | :--- | :--- |")
    
# # # # # # #     for m in METRICS_OF_INTEREST:
# # # # # # #         first = True
# # # # # # #         for k in K_VALUES:
# # # # # # #             pairs = results[m][k]
# # # # # # #             if not pairs: continue
# # # # # # #             arr_pr = np.array([p[0] for p in pairs])
# # # # # # #             arr_roc = np.array([p[1] for p in pairs])
            
# # # # # # #             ratio = np.mean((arr_pr + 1e-9)/(arr_roc + 1e-9))
# # # # # # #             win = np.sum(arr_pr > arr_roc) / len(pairs) * 100
# # # # # # #             try: 
# # # # # # #                 diff = arr_pr - arr_roc
# # # # # # #                 p_val = 1.0 if np.all(diff==0) else wilcoxon(arr_pr, arr_roc)[1]
# # # # # # #             except: p_val = 1.0
            
# # # # # # #             label = f"**{m.upper()}**" if first else ""
# # # # # # #             p_str = f"**{p_val:.2e}**" if p_val < 0.05 else f"{p_val:.3f}"
# # # # # # #             r_str = f"**{ratio:.3f}x**" if ratio > 1.0 else f"{ratio:.3f}x"
# # # # # # #             logger.log(f"| {label} | {k}% | {r_str} | {win:.1f}% | {p_str} |")
# # # # # # #             first = False
# # # # # # #         logger.log("| | | | | |")

# # # # # # # def main():
# # # # # # #     parser = argparse.ArgumentParser()
# # # # # # #     parser.add_argument("--base_dir", type=str, default="results_weighted")
# # # # # # #     parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
# # # # # # #     parser.add_argument("--model", type=str, default=None, help="Filter by model (e.g. gcn)")
# # # # # # #     parser.add_argument("--filter", type=str, default=None, help="Filter by folder name string (e.g. lr-0.0001)")
# # # # # # #     args = parser.parse_args()
    
# # # # # # #     # Filename includes filter details
# # # # # # #     output_filename = f"analysis_report_{args.dataset}"
# # # # # # #     if args.model: output_filename += f"_{args.model}"
# # # # # # #     if args.filter: output_filename += f"_{args.filter}"
# # # # # # #     output_filename += ".md"
    
# # # # # # #     out_path = os.path.join(args.base_dir, args.dataset, output_filename)
# # # # # # #     logger = ReportLogger(out_path)
    
# # # # # # #     header = f"# Analysis Report for {args.dataset}"
# # # # # # #     if args.model: header += f" ({args.model})"
# # # # # # #     if args.filter: header += f" [Filter: {args.filter}]"
# # # # # # #     logger.log(header)
    
# # # # # # #     exps = load_all_results(args.base_dir, args.dataset, args.model, args.filter)
# # # # # # #     if not exps: 
# # # # # # #         print("No results found matching your criteria.")
# # # # # # #         return
    
# # # # # # #     logger.log(f"Loaded {len(exps)} experimental runs.")
    
# # # # # # #     analyze_part_a(exps, logger)
# # # # # # #     logger.log(f"\n## Part B: The 'Ratio Test'")
# # # # # # #     perform_paired_analysis(exps, logger, None)
# # # # # # #     perform_paired_analysis(exps, logger, 'Standard')
# # # # # # #     perform_paired_analysis(exps, logger, 'Focal')
    
# # # # # # #     logger.save()

# # # # # # # if __name__ == "__main__":
# # # # # # #     main()