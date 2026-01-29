#!/bin/bash

# =============================================
# Batch script for running FSGCN experiments
# =============================================

# DATASETS=("minesweeper") #"amazon-ratings" "squirrel" "minesweeper" "tolokers" "questions" "wiki-cooc")
# DATASETS=("amazon-ratings" "squirrel" "minesweeper" "tolokers" "questions" "wiki-cooc")
DATASETS=("minesweeper")

GNNS=("fsgcn")
# METRICS=("acc")   # you fixed metric to acc here, can extend if needed
METRICS=("acc" "rocauc" "prauc" "balacc")

HIDDENS=(128)
LAYERS=(10)
LRS=(0.001)
DROPOUTS=(0.2)

EPOCHS=2000
RUNS=10

# FSGCN-specific
LAYER_NORMS=("yes")
# "no")
FEAT_TYPES=("all" "homophily" "heterophily")

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
              for LN in "${LAYER_NORMS[@]}"; do
                for FT in "${FEAT_TYPES[@]}"; do
                  echo ">>> $GNN | $DATASET | hidden=$H | layers=$L | lr=$LR | dropout=$DP | metric=$METRIC | ln=$LN | feat=$FT"
                  
                  CMD="python main_fsgcn.py \
                    --gnn $GNN \
                    --dataset $DATASET \
                    --hidden_channels $H \
                    --epochs $EPOCHS \
                    --lr $LR \
                    --runs $RUNS \
                    --fsgcn_num_layers $L \
                    --dropout $DP \
                    --metric $METRIC \
                    --fsgcn_feat_type $FT \
                    --device 4"
                  
                  # Add layer norm flag only if yes
                  if [ "$LN" == "yes" ]; then
                    CMD="$CMD --fsgcn_layer_norm"
                  fi

                  # Run
                  eval $CMD
                done
              done
            done
          done
        done
      done
    done
  done
done
