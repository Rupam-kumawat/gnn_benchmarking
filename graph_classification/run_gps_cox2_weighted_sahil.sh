#!/bin/bash

GPU_ID=3
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Running on GPU: $GPU_ID"

DATASETS=("COX2")
# MODELS=("gcn" "gat" "sage")
METRICS=("rocauc" "prauc")
LEARNING_RATES=(0.0001)
BATCH_SIZE=32
# HEADS=4
HIDDEN_DIMS=(512 256 128 64)
DROPOUTS=(0.2)
# 0.5)
LAYERS=(1 3 4 5 7)
POOLING_METHODS=("mean" "sum")
EPOCHS=300

LOSS_CONFIGS=(" " "--use_focal_loss")
LOSS_NAMES=("Standard" "FocalLoss")

# ===========================
#   REORDERED MAIN LOOP
# ===========================
for dataset in "${DATASETS[@]}"; do
    
    # dataset type
    if [ "$dataset" == "ogbg-molhiv" ]; then
      DATASET_TYPE="ogb"
    else
      DATASET_TYPE="tu"
    fi

    # for model in "${MODELS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for hidden in "${HIDDEN_DIMS[@]}"; do
          for dropout in "${DROPOUTS[@]}"; do
            for num_layers in "${LAYERS[@]}"; do
              for pool in "${POOLING_METHODS[@]}"; do

                # ------------------------------
                # metric moved *inside* here
                # ------------------------------
                for metric in "${METRICS[@]}"; do

                  # ------------- LOSS LOOP -------------
                  for i in "${!LOSS_CONFIGS[@]}"; do
                    loss_flag="${LOSS_CONFIGS[$i]}"
                    loss_name="${LOSS_NAMES[$i]}"

                    echo "----------------------------------------------------------------"
                    echo "RUNNING EXPERIMENT:"
                    echo "  Dataset: $dataset ($DATASET_TYPE)"
                    echo "  Model: $model"
                    echo "  Hidden: $hidden"
                    echo "  Dropout: $dropout"
                    echo "  Layers: $num_layers"
                    echo "  Pool: $pool"
                    echo "  Metric: $metric"
                    echo "  Loss: $loss_name"
                    echo "----------------------------------------------------------------"

                    python gps_molhiv_cox2_datasets_weighted.py \
                      --dataset "$dataset" \
                      --metric "$metric" \
                      --pool "$pool" \
                      --lr "$lr" \
                      --channels "$hidden" \
                      --num_layers "$num_layers" \
                      --epochs $EPOCHS \
                      --batch_size $BATCH_SIZE \
                      --runs 3 \
                      $loss_flag
                  done # end loss loop

                done # end metric loop

              done
            done
          done
        done
      done
    # done
done

echo "GNN tuning script finished."
