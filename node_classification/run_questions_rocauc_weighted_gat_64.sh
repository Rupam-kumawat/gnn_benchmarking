#!/bin/bash

# ==================================================
# GNN Hyperparameter Tuning: ROC-AUC (rocauc)
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
# GPU Target: 3
# Metric: rocauc
# Results Dir: results_weighted
# ==================================================

DEVICE_ID=3
# METRIC=rocauc
BASE_DIR=results_weighted
RUN_SCRIPT="python main_weighted.py"

# --- Hyperparameters from Appendix C ---
# HIDDEN=128

EPOCHS=2000
LR=0.001
RUNS=10

DROPOUTS=(0.2 0.3)
# --- Tuning Loops ---
MODELS=("gat")
#"gat")
DATASETS=("questions")
HIDDEN_DIMS=(64)
#128 256 512)
# "amazon-ratings" "squirrel" )
LAYERS="1 3 5 7 10"
# LOSS_CONFIGS=( "--use_class_weight" " " "--use_focal_loss")
LOSS_CONFIGS=( " " "--use_focal_loss")

# LOSS_NAMES=( "ClassWeighted" "FocalLoss")

LOSS_NAMES=( "Standard" "FocalLoss")


METRICS=("rocauc" "prauc")


echo "=================================================="
echo "   🚀 Starting GNN Tuning (Metric: $METRIC) 🚀"
echo "        (Running on CUDA device: $DEVICE_ID)"
echo "        (Saving to: $BASE_DIR)"
echo "=================================================="


for dataset in "${DATASETS[@]}"; do
   for DROPOUT in "${DROPOUTS[@]}"; do
    echo "DROPOUT" $DROPOUT
    echo "--- 📊 Tuning on $dataset (Metric: $METRIC) ---"
    
    for model in "${MODELS[@]}"; do
    echo "--- 🧠 Model: $model ---"

     for H in "${HIDDEN_DIMS[@]}"; do
        echo "H" $H
     
        
        
        for L in $LAYERS; do
            echo "---   Layers: $L ---"

          
            
            

              for METRIC in "${METRICS[@]}"; do


              echo "---     METRIC: $METRIC ---"



              for i in "${!LOSS_CONFIGS[@]}"; do

            
                loss_flag="${LOSS_CONFIGS[$i]}"
                loss_name="${LOSS_NAMES[$i]}"
                
                echo "---     Loss: $loss_name ---"
                
                # echo "model" $model
                # echo "LAYERS" $L

                BASE_FLAGS="--hidden_channels $H --dropout $DROPOUT --epochs $EPOCHS --lr $LR --runs $RUNS --bn --res "

                
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
      done
    done
done

echo ""
echo "=================================================="
echo "       ✅ All tuning runs completed for $METRIC. ✅"
echo "=================================================="