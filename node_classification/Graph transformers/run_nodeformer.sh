
# for layer in 1 2 3 4 5 6 7 8 9 10
# do
# for hidden_channels in 64 256 512
# do
# for dropout in 0.1 0.3 0.5 0.7
# do
# for head in 1 2 4
# do

# python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout  --num_heads $head --device $1 --model nodeformer --save_result --use_bn  --weight_decay 0.0

# python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn --weight_decay 0.0 

# python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn  --metric rocauc --weight_decay 0.0

# python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn  --metric rocauc --weight_decay 0.0

# python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn  --weight_decay 5e-5

# python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn  --weight_decay 5e-5

# python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn 

# python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn 

# python main.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nodeformer --save_result --use_bn  --weight_decay 0.0

# done 
# done 
# done 
# done



#!/bin/bash

echo "=================================================="
echo "    🚀 Starting NodeFormer Grid Search 🚀"
echo "=================================================="
echo ""

# Define the GPU device ID from the first script argument, default to 0
DEVICE=${1:-0}
echo "Running on GPU: $DEVICE"

for layer in 1 2 4 6 8
do
for hidden_channels in 64 128 256
do
for dropout in 0.1 0.3 0.5
do
for head in 1 2 4
do

echo ""
echo "--- Running with Layers: $layer, Hidden: $hidden_channels, Dropout: $dropout, Heads: $head ---"

# --- Amazon Ratings ---
python main_nodeformer.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --local_layers $layer --dropout $dropout --num_heads $head --ln --res --device $DEVICE

# --- Minesweeper ---
python main_nodeformer.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.01 --runs 3 --local_layers $layer --dropout $dropout --num_heads $head --metric rocauc --ln --res --device $DEVICE

# --- Questions ---
python main_nodeformer.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 3 --local_layers $layer --dropout $dropout --num_heads $head --metric rocauc --ln --res --device $DEVICE

# --- Tolokers ---
python main_nodeformer.py --dataset tolokers --hidden_channels $hidden_channels --epochs 1000 --lr 3e-5 --runs 3 --local_layers $layer --dropout $dropout --num_heads $head --metric rocauc --ln --res --device $DEVICE

# --- Squirrel-filtered ---
python main_nodeformer.py --dataset squirrel --hidden_channels $hidden_channels --epochs 1500 --lr 0.01 --runs 3 --local_layers $layer --dropout $dropout --num_heads $head --weight_decay 5e-4 --ln --res --device $DEVICE

# --- Wiki-cooc ---
python main_nodeformer.py --dataset wiki-cooc --hidden_channels $hidden_channels --epochs 1500 --lr 0.01 --runs 3 --local_layers $layer --dropout $dropout --num_heads $head --weight_decay 5e-4 --ln --res --device $DEVICE


done
done
done
done

echo ""
echo "=================================================="
echo "    ✅ All NodeFormer grid searches completed. ✅"
echo "=================================================="