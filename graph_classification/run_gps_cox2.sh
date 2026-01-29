#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

DATASET="COX2"
LR=0.0001
HIDDEN_DIM=64
DROPOUT=0.2
HEADS=4
EPOCHS=100
BATCH_SIZE=32
RUNS=3
METRICS=("rocauc" "prauc")
LAYERS=(5 1)
POOLING=("mean" "sum" )

for L in "${LAYERS[@]}"; do
    for P in "${POOLING[@]}"; do
        for M in "${METRICS[@]}"; do
            python gps_molhiv_cox2_datasets.py \
                --dataset $DATASET \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --channels $HIDDEN_DIM \
                --num_layers $L \
                --runs $RUNS \
                --metric $M \
                --conv_type gated \
                --pool $P
        done
    done
done
