#!/bin/bash

GPU_ID=3
echo "Running on GPU: $GPU_ID"

DATASETS=("ogbg-molhiv")
MODELS=("gcn")
METRICS=("prauc")
LEARNING_RATES=(0.0001)

HIDDEN_DIMS=(128)
DROPOUTS=(0.2)
#0.5)
LAYERS=(1 3 4 5 7) 
POOLING_METHODS=("mean" "add")

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

  for dropout in "${DROPOUTS[@]}"; do
    for model in "${MODELS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for hidden in "${HIDDEN_DIMS[@]}"; do
          
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

                    python main_graph_gnn_weighted.py \
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
                      --device $GPU_ID \
                      --epochs 100 \
                      --use_bn \
                      --use_residual \
                      --runs 3 \
                      $loss_flag

                  done # end loss loop

                done # end metric loop

              done
            done
          done
        done
      done
    done
done

echo "GNN tuning script finished."
