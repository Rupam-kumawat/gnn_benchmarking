# this remaining to run

#!/bin/bash

# ==================================================
# GNN Heterophily Experiment Runner
# ==================================================
#
# This script runs a series of GNN experiments on various
# heterophilic datasets.
#
# Usage:
#   bash run_experiments.sh [device_id]
#
# Example:
#   bash run_experiments.sh 0   (to run on GPU 0)
#   bash run_experiments.sh 1   (to run on GPU 1)
#
# ==================================================

# Check if a device ID was provided
if [ -z "$1" ]
  then
    echo "Error: No device ID provided."
    echo "Usage: bash run_experiments.sh [device_id]"
    exit 1
fi

DEVICE_ID=$1

echo "=================================================="
echo "      🚀 Starting GNN Heterophily Experiments 🚀"
echo "        (Running on CUDA device: $DEVICE_ID)"
echo "=================================================="


##
# Heterophilic Datasets
##

# --- Amazon Ratings ---
echo ""
echo "--- 📊 Running on Amazon Ratings ---"
python main.py --gnn gcn --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 4 --weight_decay 0.0 --dropout 0.5  --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 9 --weight_decay 0.0 --dropout 0.5  --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn gat --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 4 --weight_decay 0.0 --dropout 0.5  --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn gin --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 4 --weight_decay 0.0 --dropout 0.5  --bn --res --metric balacc --device $DEVICE_ID


# # --- Minesweeper ---
echo ""
echo "--- 💣 Running on Minesweeper ---"
python main.py --gnn gcn --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --metric balacc --bn --res --device $DEVICE_ID
python main.py --gnn sage --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric balacc  --bn --res --device $DEVICE_ID
python main.py --gnn gat --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric balacc  --bn --res --device $DEVICE_ID
python main.py --gnn gin --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric balacc  --bn --res --device $DEVICE_ID


# # --- Questions ---
echo ""
echo "--- ❓ Running on Questions ---"
python main.py --gnn gcn --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --metric balacc  --res --device $DEVICE_ID
python main.py --gnn sage --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 6 --weight_decay 0.0 --dropout 0.2 --metric balacc --ln --device $DEVICE_ID
python main.py --gnn gat --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric balacc  --ln --res --device $DEVICE_ID
python main.py --gnn gin --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 6 --weight_decay 0.0 --dropout 0.2 --metric balacc  --ln --res --device $DEVICE_ID


# # --- Tolokers ---
echo ""
echo "--- 👥 Running on Tolokers ---"
python main.py --gnn gcn --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric balacc --bn --res --device $DEVICE_ID
python main.py --gnn sage --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric balacc --bn --res --device $DEVICE_ID
python main.py --gnn gat --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric balacc  --bn --res --device $DEVICE_ID
python main.py --gnn gin --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric balacc  --bn --res --device $DEVICE_ID


# --- Squirrel-filtered (New) ---
echo ""
echo "--- 🐿️  Running on Squirrel-filtered ---"
python main.py --gnn gcn --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn sage --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn gat --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn gin --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID


# --- Wiki-cooc (New) ---
echo ""
echo "--- 🌐 Running on Wiki-cooc ---"
python main.py --gnn gcn --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn sage --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn gat --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID
python main.py --gnn gin --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric balacc --device $DEVICE_ID


echo ""
echo "=================================================="
echo "          ✅ All runs completed. ✅"
echo "=================================================="
