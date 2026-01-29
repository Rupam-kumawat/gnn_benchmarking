
# for layer in 1 2 3 4 5 6 7 8 9 10
# do
# for hidden_channels in 32 64 256
# do

# ## heterophilic datasets
# python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer --global_layers $layer --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --device $1 --beta 0.5  --save_result --model polynormer
# python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --device $1  --save_result--model polynormer
# python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 2 --local_layers $layer --global_layers $layer --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 8 --metric rocauc --device $1 --save_result --model polynormer
# python main.py --dataset tolokers --hidden_channels $hidden_channels --epochs 800 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.5 --in_dropout 0.2 --num_heads 16 --metric rocauc --device $1 --beta 0.1 --save_result --model polynormer
# python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.2 --global_dropout 0.5 --num_heads 8 --metric rocauc --device $1 --in_dropout 0.15 --beta 0.4 --pre_ln --save_result --model polynormer

# ## homophilic datasets
# python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1200 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $1 --save_result --model polynormer
# python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $1 --save_result --model polynormer
# python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8 --device $1 --save_result --model polynormer
# python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.1 --num_heads 8 --device $1 --save_result --model polynormer
# python main.py --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --global_layers $layer  --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device $1 --save_result --model polynormer


# done 
# done 



#!/bin/bash

echo "=================================================="
echo "     🚀 Starting Polynormer Grid Search 🚀"
echo "=================================================="
echo ""

# Define the GPU device ID from the first script argument, default to 0
DEVICE=${1:-0}
echo "Running on GPU: $DEVICE"

for layer in 1 2 4 6
do
for hidden_channels in 64 128 256
do
for dropout in 0.2 0.5
do
for head in 2 4 8
do

echo ""
echo "--- Running with Layers: $layer, Hidden: $hidden_channels, Dropout: $dropout, Heads: $head ---"

# --- Amazon Ratings ---
python main_polynormer.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --local_layers $layer --global_layers $layer --dropout $dropout --num_heads $head --device $DEVICE

# --- Minesweeper ---
python main_polynormer.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.01 --runs 3 --local_layers $layer --global_layers $layer --dropout $dropout --num_heads $head --metric rocauc --device $DEVICE

# --- Questions ---
python main_polynormer.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 3 --local_layers $layer --global_layers $layer --dropout $dropout --num_heads $head --metric rocauc --pre_ln --device $DEVICE

# --- Tolokers ---
python main_polynormer.py --dataset tolokers --hidden_channels $hidden_channels --epochs 1000 --lr 3e-5 --runs 3 --local_layers $layer --global_layers $layer --dropout $dropout --num_heads $head --metric rocauc --device $DEVICE

# --- Squirrel-filtered ---
python main_polynormer.py --dataset squirrel --hidden_channels $hidden_channels --epochs 1500 --lr 0.01 --runs 3 --local_layers $layer --global_layers $layer --dropout $dropout --num_heads $head --weight_decay 5e-4 --device $DEVICE

# --- Wiki-cooc ---
python main_polynormer.py --dataset wiki-cooc --hidden_channels $hidden_channels --epochs 1500 --lr 0.01 --runs 3 --local_layers $layer --global_layers $layer --dropout $dropout --num_heads $head --weight_decay 5e-4 --device $DEVICE


done
done
done
done

echo ""
echo "=================================================="
echo "    ✅ All Polynormer grid searches completed. ✅"
echo "=================================================="
