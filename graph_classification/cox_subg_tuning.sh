#!/bin/bash

# ==========================================================
# Grid Search: Subgraphormer on COX2 (TU Dataset)
#
# Total combinations:
# 1 (dropout) * 5 (layers) * 4 (hidden)
# * 2 (metric) * 2 (loss) * 2 (pool) = 160
# Total runs = 160 * 3 = 480
# ==========================================================

# ------------------------------
# Hyperparameter grids
# ------------------------------
DROPOUTS=(0.2)
LAYERS=(1 3 5 7 10)
HIDDEN_DIMS=(64 128 256 512)
METRICS=("acc" "balacc")
LOSS_TYPES=("standard" "focal")
POOLING_METHODS=("mean" "add")

# ------------------------------
# Fixed parameters
# ------------------------------
GPU_ID=2
SCRIPT_NAME="main_graphclass_weighted.py"

DATASET_PARAMS="--dataset COX2 --dataset_type tu"
MODEL_PARAMS="--model_name subgraphormer"
TRAINING_PARAMS="--epochs 100 --runs 3 --use_bn --use_residual"

# ------------------------------
# Bookkeeping
# ------------------------------
TOTAL_COMBOS=$((${#DROPOUTS[@]} * ${#LAYERS[@]} * ${#HIDDEN_DIMS[@]} * ${#METRICS[@]} * ${#LOSS_TYPES[@]} * ${#POOLING_METHODS[@]}))
CURRENT_RUN=0

echo "Starting Subgraphormer grid search on COX2"
echo "Total combinations: $TOTAL_COMBOS"
echo "Using GPU ID = $GPU_ID"
echo ""

# ------------------------------
# Grid search loop
# ------------------------------
for dropout in "${DROPOUTS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for hidden in "${HIDDEN_DIMS[@]}"; do
      for metric in "${METRICS[@]}"; do
        for loss in "${LOSS_TYPES[@]}"; do
          for pool in "${POOLING_METHODS[@]}"; do

            CURRENT_RUN=$((CURRENT_RUN + 1))

            echo "================================================="
            echo "RUN $CURRENT_RUN / $TOTAL_COMBOS"
            echo "Layers=$layer | Hidden=$hidden | Dropout=$dropout"
            echo "Metric=$metric | Loss=$loss | Pool=$pool"
            echo "================================================="

            # Loss flags
            if [ "$loss" == "focal" ]; then
              LOSS_FLAGS="--use_focal_loss"
            else
              LOSS_FLAGS="--use_class_weight"
            fi

            CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT_NAME \
              $DATASET_PARAMS \
              $MODEL_PARAMS \
              $TRAINING_PARAMS \
              --num_layers $layer \
              --hidden_channels $hidden \
              --dropout $dropout \
              --metric $metric \
              --pool $pool \
              $LOSS_FLAGS

            echo "--- Finished RUN $CURRENT_RUN / $TOTAL_COMBOS ---"
            echo ""

          done
        done
      done
    done
  done
done

echo "✅ Subgraphormer COX2 grid search complete."
