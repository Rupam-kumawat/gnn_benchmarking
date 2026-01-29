#!/bin/bash

# This script automates hyperparameter tuning for standard GNN models
# on specified graph classification datasets.

# --- Configuration ---
# Use the first command-line argument as the GPU ID, or default to 0
GPU_ID=${1:-0}
echo "Running on GPU: $GPU_ID"

# Define the search space
# DATASETS=("ogbg-molhiv" "COX2" "MUTAG" "COLLAB")
DATASETS=("ogbg-molhiv")

MODELS=("gcn" "gat" "sage")
#"gin")
METRICS=("rocauc" "prauc")
LEARNING_RATES=(0.001)
HIDDEN_DIMS=(128)
DROPOUTS=(0.2)
# LAYERS=(1 4 5)
LAYERS=(3 7)
POOLING_METHODS=("mean" "add")

# --- Main Tuning Loop ---
for metric in "${METRICS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    # Set dataset type based on name
    if [ "$dataset" == "ogbg-molhiv" ]; then
      DATASET_TYPE="ogb"
    else
      DATASET_TYPE="tu"
    fi

    for model in "${MODELS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for hidden in "${HIDDEN_DIMS[@]}"; do
          for dropout in "${DROPOUTS[@]}"; do
            for num_layers in "${LAYERS[@]}"; do
              for pool in "${POOLING_METHODS[@]}"; do
                
                echo "----------------------------------------------------------------"
                echo "RUNNING EXPERIMENT:"
                echo "  Dataset: $dataset ($DATASET_TYPE)"
                echo "  Model: $model (GNN)"
                echo "  Metric: $metric"
                echo "  Pool: $pool"
                echo "  Params: lr=$lr, hidden=$hidden, dropout=$dropout, layers=$num_layers"
                echo "----------------------------------------------------------------"

                # Construct and run the command
                python main_graph_gnn.py \
                  --dataset_type "$DATASET_TYPE" \
                  --dataset "$dataset" \
                  --model_family "gnn" \
                  --model_name "$model" \
                  --metric "$metric" \
                  --pool "$pool" \
                  --lr "$lr" \
                  --hidden_channels "$hidden" \
                  --dropout "$dropout" \
                  --num_layers "$num_layers" \
                  --device 4 \
                  --epochs 100 \
                  --use_bn \
                  --use_residual \
                  --runs 3
              done
            done
          done
        done
      done
    done
  done
done

echo "GNN tuning script finished."
