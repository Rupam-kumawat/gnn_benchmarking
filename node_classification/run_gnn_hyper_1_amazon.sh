# #binary datasets remaining to run

# #!/bin/bash

# # ==================================================
# # GNN Heterophily Experiment Runner
# # ==================================================
# #
# # This script runs a series of GNN experiments on various
# # heterophilic datasets.
# #
# # Usage:
# #   bash run_experiments.sh [device_id]
# #
# # Example:
# #   bash run_experiments.sh 0   (to run on GPU 0)
# #   bash run_experiments.sh 1   (to run on GPU 1)
# #
# # ==================================================

# # Check if a device ID was provided
# if [ -z "$1" ]
#   then
#     echo "Error: No device ID provided."
#     echo "Usage: bash run_experiments.sh [device_id]"
#     exit 1
# fi

# DEVICE_ID=$1

# echo "=================================================="
# echo "      🚀 Starting GNN Heterophily Experiments 🚀"
# echo "        (Running on CUDA device: $DEVICE_ID)"
# echo "=================================================="


# ##
# # Heterophilic Datasets
# ##

# # --- Amazon Ratings ---
# echo ""
# echo "--- 📊 Running on Amazon Ratings ---"
# python main.py --gnn gcn --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 4 --weight_decay 0.0 --dropout 0.5  --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 9 --weight_decay 0.0 --dropout 0.5  --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn gat --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 4 --weight_decay 0.0 --dropout 0.5  --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn gin --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 10 --local_layers 4 --weight_decay 0.0 --dropout 0.5  --bn --res --metric acc --device $DEVICE_ID


# # # --- Minesweeper ---
# echo ""
# echo "--- 💣 Running on Minesweeper ---"
# python main.py --gnn gcn --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --metric acc --bn --res --device $DEVICE_ID
# python main.py --gnn sage --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric acc  --bn --res --device $DEVICE_ID
# python main.py --gnn gat --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric acc  --bn --res --device $DEVICE_ID
# python main.py --gnn gin --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 10 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric acc  --bn --res --device $DEVICE_ID


# # # --- Questions ---
# echo ""
# echo "--- ❓ Running on Questions ---"
# python main.py --gnn gcn --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --metric acc  --res --device $DEVICE_ID
# python main.py --gnn sage --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 6 --weight_decay 0.0 --dropout 0.2 --metric acc --ln --device $DEVICE_ID
# python main.py --gnn gat --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric acc  --ln --res --device $DEVICE_ID
# python main.py --gnn gin --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 10 --local_layers 6 --weight_decay 0.0 --dropout 0.2 --metric acc  --ln --res --device $DEVICE_ID


# # # --- Tolokers ---
# echo ""
# echo "--- 👥 Running on Tolokers ---"
# python main.py --gnn gcn --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric acc --bn --res --device $DEVICE_ID
# python main.py --gnn sage --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric acc --bn --res --device $DEVICE_ID
# python main.py --gnn gat --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric acc  --bn --res --device $DEVICE_ID
# python main.py --gnn gin --dataset tolokers --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs 10 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric acc  --bn --res --device $DEVICE_ID


# # --- Squirrel-filtered (New) ---
# echo ""
# echo "--- 🐿️  Running on Squirrel-filtered ---"
# python main.py --gnn gcn --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn sage --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn gat --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn gin --dataset squirrel --hidden_channels 512 --epochs 1500 --lr 0.01 --runs 10 --local_layers 3 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID


# # --- Wiki-cooc (New) ---
# echo ""
# echo "--- 🌐 Running on Wiki-cooc ---"
# python main.py --gnn gcn --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn sage --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn gat --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID
# python main.py --gnn gin --dataset wiki-cooc --hidden_channels 128 --epochs 1500 --lr 0.01 --runs 10 --local_layers 4 --weight_decay 5e-4 --dropout 0.5 --bn --res --metric acc --device $DEVICE_ID


# echo ""
# echo "=================================================="
# echo "          ✅ All runs completed. ✅"
# echo "=================================================="







#!/bin/bash

# ==================================================
# GNN Hyperparameter Tuning Runner
# ==================================================
#
# Usage:
#   bash run_tuning.sh [device_id]
#
# Example:
#   bash run_tuning.sh 0
#
# ==================================================

# if [ -z "$1" ]; then
#     echo "Error: No device ID provided."
#     echo "Usage: bash run_tuning.sh [device_id]"
#     exit 1
# fi

# DEVICE_ID=$1

# ================================
# Define datasets and search space
# ================================

DATASETS=("amazon-ratings")
# DATASETS=("minesweeper")

# GNNS=("gcn" "sage" "gat")
# GNNS=("sage" "gat")
# METRICS=("acc" "rocauc" "prac" "balacc")

GNNS=("gcn" "sage" "gat")

GNNS=("gcn" "sage" "gat")
METRICS=("acc" "balacc" "rocauc" "prauc" )
# METRICS=("balacc")
# 
# METRICS=("prauc")

HIDDENS=(128)
# LAYERS=(1 5 10)
LAYERS=(3)
LRS=(0.001)
DROPOUTS=(0.2)
WEIGHT_DECAYS=(0)

EPOCHS=2000
RUNS=10
# bn = (1)
# res = 

# ================================
# Loop over everything
# ================================

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "=================================================="
  echo " 📊 Running on Dataset: $DATASET"
  echo "=================================================="

  for GNN in "${GNNS[@]}"; do
    for METRIC in "${METRICS[@]}"; do
      for H in "${HIDDENS[@]}"; do
        for L in "${LAYERS[@]}"; do
          for LR in "${LRS[@]}"; do
            for DP in "${DROPOUTS[@]}"; do
              for WD in "${WEIGHT_DECAYS[@]}"; do
                echo ">>> $GNN | $DATASET | hidden=$H | layers=$L | lr=$LR | dropout=$DP | wd=$WD | metric=$METRIC"
                python main.py \
                  --gnn $GNN \
                  --dataset $DATASET \
                  --hidden_channels $H \
                  --epochs $EPOCHS \
                  --lr $LR \
                  --runs $RUNS \
                  --local_layers $L \
                  --weight_decay $WD \
                  --dropout $DP \
                  --bn \
                  --res \
                  --metric $METRIC \
                  --device 3
              done
            done
          done
        done
      done
    done
  done
done
