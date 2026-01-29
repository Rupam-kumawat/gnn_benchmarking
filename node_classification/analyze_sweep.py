#!/usr/bin/env python3
"""
Analyze and aggregate results from ordinal GNN hyperparameter sweeps.
Finds best configurations for each dataset and performs
comparative analysis between GNNs, validation metrics, and loss functions.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_summary(summary_path):
    """Load summary JSON file."""
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {summary_path}: {e}")
        return None

def parse_config_from_path(path):
    """Extract configuration from result path."""
    parts = path.split('/')
    
    config = {
        'dataset': None,
        'gnn': None,
        'metric': None,
        'lr': None,
        'wd': None,
        'dropout': None,
        'hidden': None,
        'layers': None,
        'ordinal_loss': None,
        'alpha': None
    }
    
    # Extract from path: results/{dataset}/{gnn}/{params}/summary
    # Assumes path starts with 'results' or similar base_dir
    if len(parts) >= 4:
        # parts[0] is 'results', parts[1] is dataset, parts[2] is gnn, parts[3] is params
        config['dataset'] = parts[1] 
        config['gnn'] = parts[2]
        
        # Parse parameter string
        param_str = parts[3]
        for param in param_str.split('_'):
            if param.startswith('metric-'):
                config['metric'] = param.split('-')[1]
            elif param.startswith('lr-'):
                config['lr'] = float(param.split('-')[1])
            elif param.startswith('wd-'):
                config['wd'] = float(param.split('-')[1])
            elif param.startswith('do-'):
                config['dropout'] = float(param.split('-')[1])
            elif param.startswith('hid-'):
                config['hidden'] = int(param.split('-')[1])
            elif param.startswith('layers-'):
                config['layers'] = int(param.split('-')[1])
            elif param.startswith('ordinal-'):
                config['ordinal_loss'] = param.split('-')[1]
            elif param.startswith('alpha-'):
                config['alpha'] = float(param.split('-')[1])
    
    return config

def collect_all_results(base_dir='results'):
    """Collect all summary.json files from results directory."""
    results = []
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: {base_dir} directory not found!")
        return pd.DataFrame()
    
    # Find all summary.json files
    for summary_file in base_path.rglob('summary/summary.json'):
        summary = load_summary(summary_file)
        if summary is None:
            continue
            
        # --- NEW MODIFICATION: Count 'run_X' folders ---
        # Get the parent experiment directory (e.g., '.../metric-acc_lr-0.01...')
        exp_dir = summary_file.parent.parent 
        
        run_count = 0
        try:
            for item in exp_dir.iterdir():
                if item.is_dir() and item.name.startswith('run_'):
                    run_count += 1
        except Exception as e:
            print(f"Warning: Could not scan directory {exp_dir}: {e}")
            continue # Skip this summary if we can't scan its parent

        # Skip this result if the run count is not 10
        if run_count != 10:
            continue
        # --- END NEW MODIFICATION ---
            
        # Parse configuration from path
        # We need to pass the path relative to the base_dir for parsing
        relative_path_str = str(summary_file.relative_to(base_path.parent))
        config = parse_config_from_path(relative_path_str)
        
        # Skip if 'ordinal_loss' is None
        if config['ordinal_loss'] is None:
            continue
        
        # Combine config and results
        result = {**config, **summary}
        results.append(result)
    
    if not results:
        print("No ordinal results found (or none matched 10 runs)!")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def perform_comparative_analysis(df, dataset_name, gnn_list):
    """
    Performs a specific comparative analysis for a dataset,
    comparing GNNs across specified metrics and losses.
    """
    print(f"\n{'='*80}")
    print(f"Comparative Analysis for {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    dataset_df = df[df['dataset'] == dataset_name]
    
    if dataset_df.empty:
        print(f"No relevant results found for {dataset_name}")
        return

    metrics_to_compare = ['acc', 'qwk']
    losses_to_compare = ['ce', 'dwce']
    
    # This maps the validation metric to the test metric we use to rank runs
    target_metric_map = {
        'acc': 'test_accuracy_mean',
        'qwk': 'ordinal_qwk_mean'
    }

    table_data = []

    for gnn in gnn_list:
        for metric in metrics_to_compare:
            for loss in losses_to_compare:
                
                # Filter to the specific combination
                subset_df = dataset_df[
                    (dataset_df['gnn'] == gnn) &
                    (dataset_df['metric'] == metric) &
                    (dataset_df['ordinal_loss'] == loss)
                ]
                
                if subset_df.empty:
                    row_data = {
                        "GNN": gnn,
                        "Val. Metric": metric,
                        "Loss": loss,
                        "Test Acc": "N/A",
                        "Test BalAcc": "N/A",
                        "Test QWK": "N/A",
                        "Test MAE": "N/A",
                        "Params (h,l,d)": "N/A",
                        "LR": "N/A",
                        "WD": "N/A",
                    }
                    table_data.append(row_data)
                    continue

                # Find the best run for this combination
                target_col = target_metric_map[metric]
                
                # Ensure the target column exists and has values
                if target_col not in subset_df.columns or subset_df[target_col].isna().all():
                     row_data = {
                        "GNN": gnn,
                        "Val. Metric": metric,
                        "Loss": loss,
                        "Test Acc": "No Data",
                        "Test BalAcc": "No Data",
                        "Test QWK": "No Data",
                        "Test MAE": "No Data",
                        "Params (h,l,d)": "-",
                        "LR": "-",
                        "WD": "-",
                    }
                     table_data.append(row_data)
                     continue

                if 'mae' in target_col: # Lower is better for MAE
                     best_idx = subset_df[target_col].idxmin()
                else: # Higher is better for Acc, QWK
                     best_idx = subset_df[target_col].idxmax()
                
                best_row = subset_df.loc[best_idx]
                
                # Build the row for the output table
                params_str = (
                    f"h={best_row.get('hidden', '?')}, "
                    f"l={best_row.get('layers', '?')}, "
                    f"d={best_row.get('dropout', '?')}"
                )
                
                row_data = {
                    "GNN": gnn,
                    "Val. Metric": metric,
                    "Loss": loss,
                    "Test Acc": f"{best_row.get('test_accuracy_mean', 0)*100:.2f}%",
                    "Test BalAcc": f"{best_row.get('balanced_accuracy_mean', 0)*100:.2f}%",
                    "Test QWK": f"{best_row.get('ordinal_qwk_mean', 0):.4f}",
                    "Test MAE": f"{best_row.get('ordinal_mae_mean', 0):.4f}",
                    "Params (h,l,d)": params_str,
                    "LR": best_row.get('lr', 'N/A'),
                    "WD": best_row.get('wd', 'N/A'),
                }
                table_data.append(row_data)

    # Create and print the comparison table
    if not table_data:
        print("No data found to build comparison table.")
        return
        
    results_table = pd.DataFrame(table_data)
    
    # Set display options for wide tables
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000) # Set a large width
    
    print(results_table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Analyze ordinal GNN sweep results')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='Base directory containing results')
    parser.add_argument('--dataset', type=str, choices=['amazon-ratings', 'squirrel', 'both'],
                       default='both', help='Dataset to analyze')
    parser.add_argument('--gnn', type=str, choices=['gcn', 'gat', 'sage', 'fsgcn', 'all'],
                       default='all', help='GNNs to analyze (default: all)')
    parser.add_argument('--export', type=str, help='Export *filtered* results to CSV file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ORDINAL GNN HYPERPARAMETER SWEEP ANALYSIS")
    print("="*80)
    
    # --- Define GNNs and Datasets to analyze ---
    if args.gnn == 'all':
        gnn_list = ['gcn', 'gat', 'sage', 'fsgcn']
    else:
        gnn_list = [args.gnn]
        
    if args.dataset == 'both':
        dataset_list = ['amazon-ratings', 'squirrel']
    else:
        dataset_list = [args.dataset]

    # --- Collect all results ---
    print(f"\nCollecting all results from {args.base_dir}...")
    df = collect_all_results(args.base_dir)
    
    if df.empty:
        print("No results to analyze!")
        return
    
    print(f"Found {len(df)} total experiment configurations (that completed 10 runs).")

    # --- Filter down to only relevant runs ---
    df_filtered = df[
        df['dataset'].isin(dataset_list) &
        df['gnn'].isin(gnn_list) &
        df['metric'].isin(['acc', 'qwk']) &
        df['ordinal_loss'].isin(['ce', 'dwce'])
    ]
    
    if df_filtered.empty:
        print("No relevant runs found for GNNs, Datasets, Metrics, and Losses specified.")
        return
        
    print(f"Filtered down to {len(df_filtered)} relevant configurations for comparison.")
    
    # Export if requested
    if args.export:
        df_filtered.to_csv(args.export, index=False)
        print(f"Exported filtered results to {args.export}")
    
    # --- Run comparative analysis for each dataset ---
    for dataset in dataset_list:
        # Pass the filtered DF to the analysis function
        perform_comparative_analysis(df_filtered, dataset, gnn_list)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()