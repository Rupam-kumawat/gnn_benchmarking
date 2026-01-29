#!/bin/bash

# --- Configuration ---
# This script runs a grid search for ogbg-molmuv.
# Total combinations: 1 (dropout) * 1 (model) * 5 (layers) * 4 (hidden) * 1 (lr) * 2 (metrics) * 2 (loss) * 2 (pool) = 160
# Total training sessions: 160 * 3 runs = 480.

# Define the hyperparameter arrays
DROPOUTS=(0.2)
MODELS=("sage")
LAYERS=(7 5 3 1 10)
HIDDEN_DIMS=(64 128)
# HIDDEN_DIMS=(512 256 128 64)
LEARNING_RATES=(0.0001)
METRICS=("prauc" "rocauc")
LOSS_TYPES=("standard" "focal")
POOLING_METHODS=("mean" "add") # <-- ADDED pooling loop

# --- Fixed Parameters ---
GPU_ID=2
SCRIPT_NAME="main_molmuv.py"
DATASET_PARAMS="--dataset ogbg-molmuv --dataset_type ogb"
STABILITY_PARAMS="--use_bn --use_residual --pos_weight_cap 1000.0 --epochs 100 --runs 3"

# --- Grid Search Loop ---
TOTAL_COMBOS=$((${#DROPOUTS[@]} * ${#MODELS[@]} * ${#LAYERS[@]} * ${#HIDDEN_DIMS[@]} * ${#LEARNING_RATES[@]} * ${#METRICS[@]} * ${#LOSS_TYPES[@]} * ${#POOLING_METHODS[@]}))
CURRENT_RUN=0

echo "Starting grid search with $TOTAL_COMBOS total combinations..."

for dropout in "${DROPOUTS[@]}"; do
  for model in "${MODELS[@]}"; do
    for layer in "${LAYERS[@]}"; do
      for hidden in "${HIDDEN_DIMS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for metric in "${METRICS[@]}"; do
            for loss in "${LOSS_TYPES[@]}"; do
              for pool in "${POOLING_METHODS[@]}"; do # <-- Start Pooling Loop

                CURRENT_RUN=$((CURRENT_RUN + 1))
                echo "================================================================="
                echo "RUN $CURRENT_RUN / $TOTAL_COMBOS"
                echo "Params: D=$dropout, Mdl=$model, L=$layer, H=$hidden, LR=$lr, Met=$metric, Loss=$loss, Pool=$pool"
                echo "================================================================="

                # Handle the conditional loss flags
                LOSS_FLAGS=""
                if [ "$loss" == "focal" ]; then
                    LOSS_FLAGS="--use_focal_loss --focal_alpha 0.99"
                else
                    LOSS_FLAGS="--use_class_weight"
                fi

                # Construct and execute the command
                # Added --pool $pool to the command
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
                    --pool $pool \
                    $LOSS_FLAGS

                echo "--- Finished RUN $CURRENT_RUN / $TOTAL_COMBOS ---"
                echo ""

              done # End Pooling Loop
            done
          done
        done
      done
    done
  done
done

echo "All hyperparameter tuning runs complete."