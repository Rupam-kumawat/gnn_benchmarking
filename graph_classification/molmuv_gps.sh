#!/bin/bash

# --- Configuration ---
# This script runs a grid search with DROPOUT as the outermost loop.
#
# Total combinations: 2 (dropout) * 3 (models) * 5 (layers) * 4 (hidden) * 1 (lr) * 2 (metrics) * 2 (loss) = 240
# Each combination is run 3 times (as per --runs 3).
# Total training sessions: 240 * 3 = 720.

# Define the hyperparameter arrays
DROPOUTS=(0.2 0.5)      # <-- MOVED to outermost loop
MODELS=("gcn" "sage" "gat")
LAYERS=(10 7 5 3 1)
HIDDEN_DIMS=(512 256 128 64)
LEARNING_RATES=(0.0001) # <-- FIXED as requested
METRICS=("prauc" "rocauc")
LOSS_TYPES=("standard" "focal") # "standard" = BCE+ClassWeight, "focal" = FocalLoss

# --- Fixed Parameters ---
GPU_ID=3
SCRIPT_NAME="main_molmuv.py"
DATASET_PARAMS="--dataset ogbg-molmuv --dataset_type ogb"

# Parameters we found necessary for stability and performance on this dataset
STABILITY_PARAMS="--use_bn --use_residual --pos_weight_cap 1000.0 --epochs 100 --runs 3"

# --- Grid Search Loop ---
TOTAL_COMBOS=$((${#DROPOUTS[@]} * ${#MODELS[@]} * ${#LAYERS[@]} * ${#HIDDEN_DIMS[@]} * ${#LEARNING_RATES[@]} * ${#METRICS[@]} * ${#LOSS_TYPES[@]}))
CURRENT_RUN=0

echo "Starting grid search with $TOTAL_COMBOS total combinations..."

for dropout in "${DROPOUTS[@]}"; do
  for model in "${MODELS[@]}"; do
    for layer in "${LAYERS[@]}"; do
      for hidden in "${HIDDEN_DIMS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for metric in "${METRICS[@]}"; do
            for loss in "${LOSS_TYPES[@]}"; do

              CURRENT_RUN=$((CURRENT_RUN + 1))
              echo "================================================================="
              echo "RUN $CURRENT_RUN / $TOTAL_COMBOS"
              echo "Params: D=$dropout, Mdl=$model, L=$layer, H=$hidden, LR=$lr, Met=$metric, Loss=$loss"
              echo "================================================================="

              # Handle the conditional loss flags
              LOSS_FLAGS=""
              if [ "$loss" == "focal" ]; then
                  # When using focal loss for MUV, we MUST set a high alpha
                  # for the rare positive class. The default (0.25) will fail.
                  LOSS_FLAGS="--use_focal_loss --focal_alpha 0.99"
              else
                  # "standard" = BCE with positive class weighting
                  LOSS_FLAGS="--use_class_weight"
              fi

              # Construct and execute the command
              python $SCRIPT_NAME \
                  $DATASET_PARAMS \
                  $STABILITY_PARAMS \
                  --device $GPU_ID \
                  --model_name $model \
                  --num_layers $layer \
                  --hidden_channels $hidden \
                  --lr $lr \
                  --dropout $dropout \
                  --metric $metric \
                  $LOSS_FLAGS

              echo "--- Finished RUN $CURRENT_RUN / $TOTAL_COMBOS ---"
              echo ""

            done
          done
        done
      done
    done
  done
done

echo "All hyperparameter tuning runs complete."