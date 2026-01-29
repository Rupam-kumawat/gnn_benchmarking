#!/bin/bash

# ==================================================
# GNN Hyperparameter Tuning: Accuracy (acc)
# ==================================================
#
# Tunes GCN, GAT, SAGE on specified datasets.
# Tunes:
#   - Layers: 1, 3, 4, 5, 7, 10
#   - Loss: Standard, Class Weighted, Focal
#
# Script: main_weighted.py
# Params: From Appendix C (hid=128, drop=0.2, etc.)
#
# GPU Target: 1
# Metric: acc
# Results Dir: results_weighted
# ==================================================

DEVICE_ID=1
METRIC=acc
BASE_DIR=results_weighted
RUN_SCRIPT="python main_weighted.py"

# --- Hyperparameters from Appendix C ---
HIDDEN=128
DROPOUT=0.2
EPOCHS=2000
LR=0.001
RUNS=10
BASE_FLAGS="--hidden_channels $HIDDEN --epochs $EPOCHS --lr $LR --dropout $DROPOUT --runs $RUNS --bn --res"

# --- Tuning Loops ---
MODELS=("gcn" "gat" "sage")
DATASETS=("amazon-ratings" "squirrel" "questions")
LAYERS="1 3 4 5 7 10"
LOSS_CONFIGS=("--use_class_weight" "--use_focal_loss")
LOSS_NAMES=("ClassWeighted" "FocalLoss")

echo "=================================================="
echo "   🚀 Starting GNN Tuning (Metric: $METRIC) 🚀"
echo "        (Running on CUDA device: $DEVICE_ID)"
echo "        (Saving to: $BASE_DIR)"
echo "=================================================="

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--- 📊 Tuning on $dataset (Metric: $METRIC) ---"
    
    for model in "${MODELS[@]}"; do
        echo "--- 🧠 Model: $model ---"
        
        for L in $LAYERS; do
            echo "---   Layers: $L ---"
            
            for i in "${!LOSS_CONFIGS[@]}"; do
                loss_flag="${LOSS_CONFIGS[$i]}"
                loss_name="${LOSS_NAMES[$i]}"
                
                echo "---     Loss: $loss_name ---"
                
                # Run the command. The loss_flag will be empty for standard,
                # or add the correct flag for weighted/focal.
                $RUN_SCRIPT --gnn $model --dataset $dataset \
                             --local_layers $L \
                             --metric $METRIC \
                             --device $DEVICE_ID \
                             --base_results_dir $BASE_DIR \
                             $BASE_FLAGS \
                             $loss_flag
            done
        done
    done
done

echo ""
echo "=================================================="
echo "       ✅ All tuning runs completed for $METRIC. ✅"
echo "=================================================="