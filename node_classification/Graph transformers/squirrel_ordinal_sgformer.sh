#!/bin/bash
# ========================================================================
# NodeFormer Squirrel Experiment Sweep Script
# GPU: 3
# Loops: Dropout > HiddenDim > Layers > OrdinalLoss > DWCE_Alpha > Metric
# Fixed: LR=0.01, WeightDecay=0.0005, Flags: --ln --res
# Auto-saves results to unique subfolders
# ========================================================================

GPU_ID=0
DATASET="squirrel"
MODEL="sgformer"
LR=0.01
WEIGHT_DECAY=0.0005
EPOCHS=1500
RUNS=10

echo "Running NodeFormer ordinal experiments on GPU ${GPU_ID}"
echo "Dataset: ${DATASET}"

# Parameter grids
DROPOUTS=(0.5 0.2)
HIDDENS=(64 256 128 512)
LAYERS=(1 3 5 7 10)
ORDINAL_LOSSES=("ce" "dwce")
DWCE_ALPHAS=(0.1 0.5 2.0 5.0)
METRICS=("acc" "qwk")

# Create main results directory
BASE_DIR="results/${DATASET}/${MODEL}"
mkdir -p $BASE_DIR

# Iterate loops: dropout > hidden_dim > layers > ordinal_loss > dwce_alpha > metric
for DROPOUT in "${DROPOUTS[@]}"; do
  for HIDDEN in "${HIDDENS[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
      for LOSS in "${ORDINAL_LOSSES[@]}"; do
        if [ "$LOSS" == "dwce" ]; then
          for ALPHA in "${DWCE_ALPHAS[@]}"; do
            for METRIC in "${METRICS[@]}"; do
              
              # Generate result directory
              OUT_DIR="${BASE_DIR}/dropout_${DROPOUT}_dim_${HIDDEN}_layers_${LAYER}_loss_${LOSS}_alpha_${ALPHA}_metric_${METRIC}"
              mkdir -p "$OUT_DIR"
              
              echo ">>> RUN: dropout=$DROPOUT, hidden=$HIDDEN, layers=$LAYER, loss=$LOSS(alpha=$ALPHA), metric=$METRIC"
              echo "Saving results to: $OUT_DIR"
              
              python main_sgformer_ordinal.py \
                --dataset $DATASET \
                --gnn $MODEL \
                --local_layers $LAYER \
                --hidden_channels $HIDDEN \
                --dropout $DROPOUT \
                --lr $LR \
                --weight_decay $WEIGHT_DECAY \
                --epochs $EPOCHS \
                --runs $RUNS \
                --metric $METRIC \
                --ordinal_loss $LOSS \
                --dwce_alpha $ALPHA \
                --device $GPU_ID \
                --ln \
                --res \
                --output_dir "$OUT_DIR" > "${OUT_DIR}/log.txt" 2>&1
              
              echo "-------------------------------------------------------------"
            done
          done
        else
          for METRIC in "${METRICS[@]}"; do
            OUT_DIR="${BASE_DIR}/dropout_${DROPOUT}_dim_${HIDDEN}_layers_${LAYER}_loss_${LOSS}_metric_${METRIC}"
            mkdir -p "$OUT_DIR"
            
            echo ">>> RUN: dropout=$DROPOUT, hidden=$HIDDEN, layers=$LAYER, loss=$LOSS, metric=$METRIC"
            echo "Saving results to: $OUT_DIR"
            
            python main_sgformer_ordinal.py \
              --dataset $DATASET \
              --gnn $MODEL \
              --local_layers $LAYER \
              --hidden_channels $HIDDEN \
              --dropout $DROPOUT \
              --lr $LR \
              --weight_decay $WEIGHT_DECAY \
              --epochs $EPOCHS \
              --runs $RUNS \
              --metric $METRIC \
              --ordinal_loss $LOSS \
              --device $GPU_ID \
              --ln \
              --res \
              --output_dir "$OUT_DIR" > "${OUT_DIR}/log.txt" 2>&1
            
            echo "-------------------------------------------------------------"
          done
        fi
      done
    done
  done
done

echo "✅ All experiments completed."
