#!/bin.bash

export CUDA_VISIBLE_DEVICES=1
echo "Running on GPU: 1"

DATASET="ogbg-molhiv"
LR=0.0001
HIDDEN_DIM=64
DROPOUT=0.2
HEADS=4
EPOCHS=100
BATCH_SIZE=32
RUNS=3
METRICS=("rocauc" "prauc")
LAYERS=(10 5 1)
POOLING=("mean" "sum")

# --- ADDED: Loss Configuration Loop ---
LOSS_CONFIGS=(
    "--use_class_weight"
    "--use_focal_loss"
)

# Loop over loss configs
for loss_flags in "${LOSS_CONFIGS[@]}"; do
    echo "================================================================"
    if [ -z "$loss_flags" ]; then
        echo "LOSS CONFIG: Standard"
    else
        echo "LOSS CONFIG: $loss_flags"
    fi
    echo "================================================================"

    for L in "${LAYERS[@]}"; do
        for P in "${POOLING[@]}"; do
            for M in "${METRICS[@]}"; do

                echo "---"
                echo "Starting run: Metric=$M, Layers=$L, Pool=$P, Loss='$loss_flags'"
                echo "---"

                python gps_molhiv_cox2_datasets_weighted.py \
                    --dataset $DATASET \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --lr $LR \
                    --channels $HIDDEN_DIM \
                    --num_layers $L \
                    --runs $RUNS \
                    --metric $M \
                    --conv_type gated \
                    --pool $P \
                    $loss_flags
            done
        done
    done
done

echo "All tuning runs for ogbg-molhiv finished."