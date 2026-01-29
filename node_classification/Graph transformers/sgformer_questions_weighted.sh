#!/bin/bash
# ========================================================================
# SGFormer Questions Experiment Sweep
# GPU: 1 (Change as needed)
# Dataset: questions
# Sweep: HiddenDim (64,128,256,512) x Layers (1,3,5,7,10) x Loss (standard, focal)
# Fixed: Dropout=0.2, Metric sweep: (rocauc, prauc)
# ========================================================================

GPU_ID=3
DATASET="questions"
MODEL="sgformer"
LR=0.001
WEIGHT_DECAY=0.0005
EPOCHS=1000
RUNS=10
DROPOUT=0.2

# Parameter grids
HIDDENS=(64 128 256 512)
LAYERS=(1 3 5 7 10)
LOSSES=("standard" "focal")
METRICS=("prauc")
FOCAL_GAMMA=2.0

BASE_DIR="results/${DATASET}/${MODEL}"
mkdir -p $BASE_DIR

for HIDDEN in "${HIDDENS[@]}"; do
  for LAYER in "${LAYERS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      for METRIC in "${METRICS[@]}"; do
        
        # Directory naming logic
        if [ "$LOSS" == "focal" ]; then
          OUT_DIR="${BASE_DIR}/dim_${HIDDEN}_layers_${LAYER}_loss_focal_gamma_${FOCAL_GAMMA}_metric_${METRIC}"
          FOCAL_FLAG="--use_focal --focal_gamma ${FOCAL_GAMMA}"
        else
          OUT_DIR="${BASE_DIR}/dim_${HIDDEN}_layers_${LAYER}_loss_standard_metric_${METRIC}"
          FOCAL_FLAG=""
        fi

        mkdir -p "$OUT_DIR"
        
        echo ">>> SGFORMER RUN: dim=$HIDDEN, layers=$LAYER, loss=$LOSS, metric=$METRIC"
        
        python main_sgformer_weighted.py \
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
          --device $GPU_ID \
          --ln --res \
          $FOCAL_FLAG \
          --output_dir "$OUT_DIR" > "${OUT_DIR}/log.txt" 2>&1
        
        echo "Done. Results in $OUT_DIR"
        echo "-------------------------------------------------------------"
      done
    done
  done
done

echo "✅ SGFormer sweep completed."