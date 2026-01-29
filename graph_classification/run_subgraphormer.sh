#!/bin/bash

# This script automates hyperparameter tuning for Graph Transformer models
# on specified graph classification datasets.

# --- Configuration ---
# Use the first command-line argument as the GPU ID, or default to 0
# GPU_ID=${1:-0}
# echo "Running on GPU: $GPU_ID"

# Define the search space
# DATASETS=("ogbg-molhiv" "COX2" "MUTAG" "COLLAB")
DATASETS=("ogbg-molhiv" "COX2" "MUTAG")
# MODELS=("gps" "subgraphormer" "graphvit")
MODELS=("subgraphormer")

#"graphormer" "graphit")
METRICS=("rocauc" "prauc")
LEARNING_RATES=(0.001) # Transformers often benefit from smaller LRs
HIDDEN_DIMS=(128)
DROPOUTS=(0.2)
LAYERS=(5)
#2)
HEADS=(4)
POOLING_METHODS=("mean" "add")
# POOLING_METHODS=("mean")
# "add")

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
              for num_heads in "${HEADS[@]}"; do
                for pool in "${POOLING_METHODS[@]}"; do

                  echo "----------------------------------------------------------------"
                  echo "RUNNING EXPERIMENT:"
                  echo "  Dataset: $dataset ($DATASET_TYPE)"
                  echo "  Model: $model (Graph Transformer)"
                  echo "  Metric: $metric"
                  echo "  Pool: $pool"
                  echo "  Params: lr=$lr, hidden=$hidden, dropout=$dropout, layers=$num_layers, heads=$num_heads"
                  echo "----------------------------------------------------------------"

                  # Construct and run the command
                  python main_graphclass.py \
                    --dataset_type "$DATASET_TYPE" \
                    --dataset "$dataset" \
                    --model_family "gt" \
                    --model_name "$model" \
                    --metric "$metric" \
                    --pool "$pool" \
                    --lr "$lr" \
                    --hidden_channels "$hidden" \
                    --dropout "$dropout" \
                    --num_layers "$num_layers" \
                    --nhead "$num_heads" \
                    --device 3 \
                    --epochs 100 \
                    --runs 3
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Graph Transformer tuning script finished."
